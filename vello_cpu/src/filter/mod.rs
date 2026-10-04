// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Filter effects implementation for `vello_cpu`.
//!
//! This module provides CPU-based implementations of SVG filter effects,
//! supporting both low-precision (u8) and high-precision (f32) rendering paths.
//! Filters are applied to rendered layer pixmaps and may use scratch storage for
//! intermediate buffers.

mod color_matrix;
pub(crate) mod context;
mod drop_shadow;
mod flood;
mod gaussian_blur;
mod offset;
mod shift;

use alloc::vec::Vec;
use context::ScratchBuffer;
use vello_common::filter::PreparedFilter;
use vello_common::filter_effects::Filter;
use vello_common::kurbo::Affine;
use vello_common::pixmap::Pixmap;

/// Trait for filter effects that can be applied to layers.
///
/// Each filter implements this trait with both u8 and f32 variants
/// to support different rendering backends and precision requirements.
/// The low-precision path (`execute_lowp`) uses 8-bit color channels for
/// better performance, while the high-precision path (`execute_highp`)
/// uses 32-bit floating-point for higher quality at the cost of speed.
pub(crate) trait FilterEffect {
    /// Apply the low-precision (u8) version of the filter.
    ///
    /// # Arguments
    /// * `pixmap` - The target pixmap containing rendering metadata
    /// * `filter_scratch` - Reusable scratch storage for intermediate buffers
    fn execute_lowp(&self, pixmap: &mut Pixmap, filter_scratch: &mut ScratchBuffer);

    /// Apply the high-precision (f32) version of the filter.
    ///
    /// # Arguments
    /// * `pixmap` - The target pixmap containing rendering metadata
    /// * `filter_scratch` - Reusable scratch storage for intermediate buffers
    fn execute_highp(&self, pixmap: &mut Pixmap, filter_scratch: &mut ScratchBuffer);
}

/// Apply the low-precision (u8) version of a filter effect to a layer.
///
/// This function dispatches filter primitives from a filter graph to their
/// corresponding CPU implementations using 8-bit color channels.
///
/// # Arguments
/// * `filter` - The filter containing the graph of primitives to apply
/// * `pixmap` - The target pixmap containing rendering metadata
/// * `filter_scratch` - Reusable scratch storage for intermediate buffers
/// * `transform` - The transformation matrix to extract scale from for filter parameters
///
/// See [`apply_chain`] for how the primitives of the graph are applied.
pub(crate) fn filter_lowp(
    filter: &Filter,
    pixmap: &mut Pixmap,
    filter_scratch: &mut ScratchBuffer,
    transform: Affine,
) {
    apply_chain(filter, pixmap, filter_scratch, transform, Precision::Low);
}

/// Apply the high-precision (f32) version of a filter effect to a layer.
///
/// This function dispatches filter primitives from a filter graph to their
/// corresponding CPU implementations using 32-bit floating-point color channels.
///
/// # Arguments
/// * `filter` - The filter containing the graph of primitives to apply
/// * `pixmap` - The target pixmap containing rendering metadata
/// * `filter_scratch` - Reusable scratch storage for intermediate buffers
/// * `transform` - The transformation matrix to extract scale from for filter parameters
///
/// See [`apply_chain`] for how the primitives of the graph are applied.
pub(crate) fn filter_highp(
    filter: &Filter,
    pixmap: &mut Pixmap,
    filter_scratch: &mut ScratchBuffer,
    transform: Affine,
) {
    apply_chain(filter, pixmap, filter_scratch, transform, Precision::High);
}

#[derive(Clone, Copy)]
enum Precision {
    Low,
    High,
}

/// Apply the primitives of a filter graph one after another (see
/// [`FilterGraph`](vello_common::filter_effects::FilterGraph)).
///
/// Consecutive color matrices are applied in one pass, without rounding their intermediate
/// results to 8 bits. A graph with a primitive that is not implemented yet leaves the layer
/// unfiltered.
fn apply_chain(
    filter: &Filter,
    pixmap: &mut Pixmap,
    filter_scratch: &mut ScratchBuffer,
    transform: Affine,
    precision: Precision,
) {
    fn execute(
        effect: &impl FilterEffect,
        pixmap: &mut Pixmap,
        filter_scratch: &mut ScratchBuffer,
        precision: Precision,
    ) {
        match precision {
            Precision::Low => effect.execute_lowp(pixmap, filter_scratch),
            Precision::High => effect.execute_highp(pixmap, filter_scratch),
        }
    }

    let chain = PreparedFilter::chain(filter, &transform).unwrap_or_default();
    let mut matrices: Vec<[f32; 20]> = Vec::new();
    for prepared_filter in chain {
        if let PreparedFilter::ColorMatrix(color_matrix) = prepared_filter {
            matrices.push(color_matrix.matrix);
            continue;
        }
        color_matrix::apply_color_matrices(pixmap, &matrices);
        matrices.clear();
        match prepared_filter {
            PreparedFilter::Flood(flood) => execute(&flood, pixmap, filter_scratch, precision),
            PreparedFilter::GaussianBlur(blur) => {
                execute(&blur, pixmap, filter_scratch, precision);
            }
            PreparedFilter::Offset(offset) => execute(&offset, pixmap, filter_scratch, precision),
            PreparedFilter::DropShadow(drop_shadow) => {
                execute(&drop_shadow, pixmap, filter_scratch, precision);
            }
            PreparedFilter::ColorMatrix(_) => unreachable!("collected above"),
        }
    }
    color_matrix::apply_color_matrices(pixmap, &matrices);
}

#[cfg(test)]
mod chain_tests {
    use crate::{RenderContext, Resources};
    use vello_common::color::palette::css::WHITE;
    use vello_common::color::{AlphaColor, PremulRgba8, Srgb};
    use vello_common::filter_effects::{EdgeMode, Filter, FilterPrimitive};
    use vello_common::kurbo::Rect;
    use vello_common::pixmap::Pixmap;

    fn render(filter: Filter) -> Pixmap {
        let mut ctx = RenderContext::new(64, 64);
        ctx.set_paint(WHITE);
        ctx.fill_rect(&Rect::new(0.0, 0.0, 64.0, 64.0));
        ctx.push_filter_layer(filter);
        ctx.set_paint(AlphaColor::<Srgb>::from_rgb8(200, 100, 50));
        ctx.fill_rect(&Rect::new(16.0, 16.0, 48.0, 48.0));
        ctx.pop_layer();
        ctx.flush();
        let mut pixmap = Pixmap::new(64, 64);
        ctx.render(pixmap.as_mut(), &mut Resources::new());
        pixmap
    }

    const fn scale(amount: f32) -> FilterPrimitive {
        FilterPrimitive::ColorMatrix {
            matrix: [
                amount, 0.0, 0.0, 0.0, 0.0, //
                0.0, amount, 0.0, 0.0, 0.0, //
                0.0, 0.0, amount, 0.0, 0.0, //
                0.0, 0.0, 0.0, 1.0, 0.0,
            ],
        }
    }

    const INVERT: FilterPrimitive = FilterPrimitive::ColorMatrix {
        matrix: [
            -1.0, 0.0, 0.0, 0.0, 1.0, //
            0.0, -1.0, 0.0, 0.0, 1.0, //
            0.0, 0.0, -1.0, 0.0, 1.0, //
            0.0, 0.0, 0.0, 1.0, 0.0,
        ],
    };

    #[test]
    fn primitives_apply_in_order() {
        // (200, 100, 50) halved, then inverted: (155, 205, 230).
        let pixmap = render(Filter::from_primitives([scale(0.5), INVERT]));
        assert_eq!(
            pixmap.sample(32, 32),
            PremulRgba8 {
                r: 155,
                g: 205,
                b: 230,
                a: 255
            }
        );
        // Inverted, then halved: (27.5, 77.5, 102.5), computed in f32 without rounding the
        // inverted color to 8 bits first (Chromium draws (27, 77, 102)).
        let pixmap = render(Filter::from_primitives([INVERT, scale(0.5)]));
        assert_eq!(
            pixmap.sample(32, 32),
            PremulRgba8 {
                r: 27,
                g: 77,
                b: 103,
                a: 255
            }
        );
    }

    #[test]
    fn chained_offsets_are_not_cut_off() {
        // Two offsets move the square by 12px in total; the layer must be large enough.
        let offset = FilterPrimitive::Offset { dx: 6.0, dy: 0.0 };
        let pixmap = render(Filter::from_primitives([offset.clone(), offset]));
        assert_eq!(pixmap.sample(20, 32), PremulRgba8::from_u32(u32::MAX));
        assert_eq!(pixmap.sample(58, 32).r, 200);
        let blur = FilterPrimitive::GaussianBlur {
            std_deviation: 2.0,
            edge_mode: EdgeMode::None,
        };
        // A blur after a colour matrix still blurs the matrix's output.
        let pixmap = render(Filter::from_primitives([INVERT, blur]));
        let edge = pixmap.sample(16, 32);
        assert!(edge.r > 55 && edge.r < 255, "{edge:?}");
    }

    #[test]
    fn unsupported_primitives_leave_the_layer_unfiltered() {
        let pixmap = render(Filter::from_primitives([scale(0.5), FilterPrimitive::Tile]));
        assert_eq!(pixmap.sample(32, 32).r, 200);
    }
}
