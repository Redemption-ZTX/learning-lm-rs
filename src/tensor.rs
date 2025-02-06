use half::f16;
use std::{slice, sync::Arc, vec};

#[derive(Debug, Clone)]
pub enum TensorType {
    F16,
    F32,
}

pub trait NumType: Copy + Clone + Default + Send + Sync + 'static {
    fn to_f32(self) -> f32;
    fn from_f32(x: f32) -> Self;
}

impl NumType for f32 {
    fn to_f32(self) -> f32 {
        self
    }
    fn from_f32(x: f32) -> Self {
        x
    }
}

impl NumType for f16 {
    // 需要添加 half crate 依赖
    fn to_f32(self) -> f32 {
        self.to_f32()
    }
    fn from_f32(x: f32) -> Self {
        f16::from_f32(x)
    }
}

impl NumType for i8 {
    fn to_f32(self) -> f32 {
        self as f32
    }

    fn from_f32(x: f32) -> Self {
        x.round().clamp(-127.0, 127.0) as i8
    }
}

#[derive(Debug, Clone)]
pub struct Tensor<T> {
    dtype: TensorType,
    data: Arc<Box<[T]>>,
    shape: Vec<usize>,
    offset: usize,
    length: usize,
}

impl<T: Copy + Clone + Default + 'static> Tensor<T> {
    pub fn new(data: Vec<T>, shape: &Vec<usize>) -> Self {
        let length = data.len();
        let dtype = if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f16>() {
            TensorType::F16
        } else {
            TensorType::F32
        };
        Tensor {
            dtype,
            data: Arc::new(data.into_boxed_slice().try_into().unwrap()),
            shape: shape.clone(),
            offset: 0,
            length: length,
        }
    }

    pub fn default(shape: &Vec<usize>) -> Self {
        let length = shape.iter().product();
        let data = vec![T::default(); length];
        Self::new(data, shape)
    }

    pub fn dtype(&self) -> &TensorType {
        &self.dtype
    }

    pub fn data(&self) -> &[T] {
        &self.data[self.offset..][..self.length]
    }

    pub unsafe fn data_mut(&mut self) -> &mut [T] {
        let ptr = self.data.as_ptr().add(self.offset) as *mut T;
        slice::from_raw_parts_mut(ptr, self.length)
    }

    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    pub fn size(&self) -> usize {
        self.length
    }

    // Reinterpret the tensor as a new shape while preserving total size.
    pub fn reshape(&mut self, new_shape: &Vec<usize>) -> &mut Self {
        let new_length: usize = new_shape.iter().product();
        if new_length != self.length {
            let old_shape = self.shape.clone();
            panic!("New shape {new_shape:?} does not match tensor of {old_shape:?}");
        }
        self.shape = new_shape.clone();
        self
    }

    pub fn slice(&self, start: usize, shape: &Vec<usize>) -> Self {
        let new_length: usize = shape.iter().product();
        assert!(self.offset + start + new_length <= self.length);
        Tensor {
            dtype: self.dtype.clone(),
            data: self.data.clone(),
            shape: shape.clone(),
            offset: self.offset + start,
            length: new_length,
        }
    }

    pub fn to_f16(&self) -> Tensor<f16>
    where
        T: Into<f16> + Copy,
    {
        let data: Vec<f16> = self.data().iter().map(|&x| x.into()).collect();
        Tensor::new(data, self.shape())
    }

    pub fn to_f32(&self) -> Tensor<f32>
    where
        T: Into<f32> + Copy,
    {
        let data: Vec<f32> = self.data().iter().map(|&x| x.into()).collect();
        Tensor::new(data, self.shape())
    }
}

// /Some helper functions for testing and debugging
impl Tensor<f32> {
    #[allow(unused)]
    pub fn close_to(&self, other: &Self, rel: f32) -> bool {
        if self.shape() != other.shape() {
            return false;
        }
        let a = self.data();
        let b = other.data();

        return a.iter().zip(b).all(|(x, y)| float_eq(x, y, rel));
    }
    #[allow(unused)]
    pub fn print(&self) {
        println!(
            "shpae: {:?}, offset: {}, length: {}",
            self.shape, self.offset, self.length
        );
        let dim = self.shape()[self.shape().len() - 1];
        let batch = self.length / dim;
        for i in 0..batch {
            let start = i * dim;
            println!("{:?}", &self.data()[start..][..dim]);
        }
    }
}

#[inline]
pub fn float_eq(x: &f32, y: &f32, rel: f32) -> bool {
    (x - y).abs() <= rel * (x.abs() + y.abs()) / 2.0
}

// Implementation for f16 tensors
impl Tensor<f16> {
    #[allow(unused)]
    pub fn close_to(&self, other: &Self, rel: f16) -> bool {
        if self.shape() != other.shape() {
            return false;
        }
        let a = self.data();
        let b = other.data();

        return a.iter().zip(b).all(|(x, y)| {
            let x_f32 = f32::from(*x);
            let y_f32 = f32::from(*y);
            float_eq(&x_f32, &y_f32, f32::from(rel))
        });
    }

    #[allow(unused)]
    pub fn print(&self) {
        println!(
            "shape: {:?}, offset: {}, length: {}",
            self.shape, self.offset, self.length
        );
        let dim = self.shape()[self.shape().len() - 1];
        let batch = self.length / dim;
        for i in 0..batch {
            let start = i * dim;
            println!(
                "{:?}",
                &self.data()[start..][..dim]
                    .iter()
                    .map(|x| f32::from(*x))
                    .collect::<Vec<f32>>()
            );
        }
    }
}
