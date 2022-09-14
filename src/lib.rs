#![feature(portable_simd)]
#![feature(test)]

extern crate test;

use std::simd::Simd;

trait Svf {
    fn new(cutoff: f32, res: f32, sample_rate: f32) -> Self;
    fn set(&mut self, cutoff: f32, res: f32, sample_rate: f32);
    fn reset(&mut self);
    fn process(&mut self, input: f32) -> f32;
}

pub struct Scalar {
    a1: f32,
    a2: f32,
    a3: f32,
    ic1eq: f32,
    ic2eq: f32,
}

impl Svf for Scalar {
    #[inline]
    fn new(cutoff: f32, res: f32, sample_rate: f32) -> Scalar {
        let mut svf = Scalar {
            a1: 0.0,
            a2: 0.0,
            a3: 0.0,
            ic1eq: 0.0,
            ic2eq: 0.0,
        };

        svf.set(cutoff, res, sample_rate);

        svf
    }

    #[inline]
    fn set(&mut self, cutoff: f32, res: f32, sample_rate: f32) {
        let g = (std::f32::consts::PI * (cutoff / sample_rate)).tan();
        let k = 2.0 - 2.0 * res;

        self.a1 = 1.0 / (1.0 + g * (g + k));
        self.a2 = g * self.a1;
        self.a3 = g * self.a2;
    }

    #[inline]
    fn reset(&mut self) {
        self.ic1eq = 0.0;
        self.ic2eq = 0.0;
    }

    #[inline]
    fn process(&mut self, input: f32) -> f32 {
        let v3 = input - self.ic2eq;
        let v1 = self.a1 * self.ic1eq + self.a3 * self.ic2eq;
        let v2 = self.ic2eq + self.a2 * self.ic1eq + self.a3 * v3;
        self.ic1eq = 2.0 * v1 - self.ic1eq;
        self.ic2eq = 2.0 * v2 - self.ic2eq;

        v2
    }
}

pub struct Mat3x3 {
    col0: Simd<f32, 4>,
    col1: Simd<f32, 4>,
    col2: Simd<f32, 4>,
    x0: f32,
    x1: f32,
}

impl Svf for Mat3x3 {
    #[inline]
    fn new(cutoff: f32, res: f32, sample_rate: f32) -> Mat3x3 {
        let mut svf = Mat3x3 {
            col0: Simd::splat(0.0),
            col1: Simd::splat(0.0),
            col2: Simd::splat(0.0),
            x0: 0.0,
            x1: 0.0,
        };

        svf.set(cutoff, res, sample_rate);

        svf
    }

    #[inline]
    fn set(&mut self, cutoff: f32, res: f32, sample_rate: f32) {
        let g = (std::f32::consts::PI * (cutoff / sample_rate)).tan();
        let k = 2.0 - 2.0 * res;

        let a1 = 1.0 / (1.0 + g * (g + k));
        let a2 = g * a1;
        let a3 = g * a2;

        self.col0 = Simd::from([(2.0 * a1 - 1.0), 2.0 * a2, a2, 0.0]);
        self.col1 = Simd::from([2.0 * a3, 1.0 - 2.0 * a3, 1.0 - a3, 0.0]);
        self.col2 = Simd::from([0.0, 2.0 * a3, a3, 0.0]);
    }

    #[inline]
    fn reset(&mut self) {
        self.x0 = 0.0;
        self.x1 = 0.0;
    }

    #[inline]
    fn process(&mut self, input: f32) -> f32 {
        let out = self.col0 * Simd::splat(self.x0)
            + self.col1 * Simd::splat(self.x1)
            + self.col2 * Simd::splat(input);
        self.x0 = out[0];
        self.x1 = out[1];
        out[2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;

    const RES: f32 = 0.5;
    const SAMPLE_RATE: f32 = 48_000.0;

    fn run_static<S: Svf>(b: &mut Bencher) {
        let input = (0..)
            .map(|x| (x as f32).sin())
            .take(128)
            .collect::<Vec<f32>>();
        let mut output = [0.0; 128];

        b.iter(|| {
            let mut svf = S::new(10_000.0, RES, SAMPLE_RATE);
            for (i, o) in input.iter().zip(output.iter_mut()) {
                *o = svf.process(*i);
            }
        });
    }

    fn run_dynamic<S: Svf>(b: &mut Bencher) {
        let input = (0..)
            .map(|x| (x as f32).sin())
            .take(128)
            .collect::<Vec<f32>>();
        let mut output = [0.0; 128];
        let cutoffs = (0..)
            .map(|x| 20_000.0 * (x as f32).sin())
            .take(128)
            .collect::<Vec<f32>>();

        b.iter(|| {
            let mut svf = S::new(cutoffs[0], RES, SAMPLE_RATE);
            for ((i, o), f) in input.iter().zip(output.iter_mut()).zip(cutoffs.iter()) {
                svf.set(*f, RES, SAMPLE_RATE);
                *o = svf.process(*i);
            }
        });
    }

    #[bench]
    fn scalar_static(b: &mut Bencher) {
        run_static::<Scalar>(b);
    }

    #[bench]
    fn matrix_3x3_static(b: &mut Bencher) {
        run_static::<Mat3x3>(b);
    }

    #[bench]
    fn scalar_dynamic(b: &mut Bencher) {
        run_dynamic::<Scalar>(b);
    }

    #[bench]
    fn matrix_3x3_dynamic(b: &mut Bencher) {
        run_dynamic::<Mat3x3>(b);
    }
}
