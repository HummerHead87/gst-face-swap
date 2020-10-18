use dlib_face_recognition as dlib;
use dlib_face_recognition::LandmarkPredictorTrait;

use anyhow::Error;
use opencv::prelude::*;
use opencv::{core, core::Mat, types};

pub fn detect_landmarks(
    predictor: &dlib::LandmarkPredictor,
    img: &Mat,
    face: &core::Rect,
) -> Result<types::VectorOfPoint2f, Error> {
    let img_matrix =
        unsafe { dlib::ImageMatrix::new(img.cols() as usize, img.rows() as usize, img.data()?) };

    let mut rect = dlib::Rectangle::default();
    rect.top = face.y as i64;
    rect.left = face.x as i64;
    rect.bottom = (face.y + face.height) as i64;
    rect.right = (face.x + face.width) as i64;
    let landmarks = predictor.face_landmarks(&img_matrix, &rect);

    Ok(types::VectorOfPoint2f::from_iter(
        landmarks
            .iter()
            .map(|p| core::Point2f::new(p.x() as f32, p.y() as f32)),
    ))
}
