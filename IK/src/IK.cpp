#include "IK.h"

IK::IK() {
  marker.resize(19);
  angleResults.resize(19);
  for (int i = 0; i < 19; ++i) {
    angleResults[i] = 0;
  }
}

void IK::DoIKTest() {
  Eigen::VectorXd angles(7);
  angles(0) = 10;
  angles(1) = 10;
  angles(2) = 10;
  angles(3) = 10;
  angles(4) = 0;
  angles(5) = 0;
  angles(6) = 0;
  Point3DVector points;

  //root marker
  points.push_back(Eigen::Vector3d(0, 0, 0));

  Eigen::Vector3d shift(2, 1, 0); //for index
  //points.push_back(Eigen::Vector3d(0.0, 0, 0.0) + shift);
  points.push_back(Eigen::Vector3d(0.0, 0.5, 0.2) + shift);
  points.push_back(Eigen::Vector3d(0.0, 1.5, 0.2) + shift);
  points.push_back(Eigen::Vector3d(0.0, 2.5, 0.2) + shift);

  //points.push_back(Eigen::Vector3d(0, 0, 0));
  HandFunctorNumericalDiff functor;
  functor.Points = points;

  Eigen::LevenbergMarquardt<HandFunctorNumericalDiff> lm(functor);

  Eigen::LevenbergMarquardtSpace::Status status = lm.minimize(angles);
  std::cout << "status: " << status << std::endl;
  std::cout << "x that minimizes the function: " << std::endl << angles << std::endl;
}


void IK::setData(const std::vector<Marker>& data) {
  //Improve sorting
  for (int i = 0; i < 19; ++i) {
    for (int j = 0; j < 19; ++j) {
      if (data[j].label == marker_label(i)) {
        marker[i] = data[j].pos;
        continue;
      }
    }
  }
}

std::vector<Eigen::Vector3f> IK::GetMarkerData() {
  return marker;
}

Eigen::VectorXd IK::DoIKRoot() {
  //CenterAround(16);
  Eigen::VectorXd angles(4);
  angles.fill(0.0f);
  Point3DVector points;
  points.push_back(marker[16].cast<double>());
  points.push_back(marker[17].cast<double>());
  points.push_back(marker[18].cast<double>());
  points.push_back(marker[4].cast<double>());

  HandFunctorNumericalDiff functor;
  functor.Points = points;

  Eigen::LevenbergMarquardt<HandFunctorNumericalDiff> lm(functor);

  Eigen::LevenbergMarquardtSpace::Status status = lm.minimize(angles);
  std::cout << "status: " << status << std::endl;
  std::cout << "x that minimizes the function: " << std::endl << angles << std::endl;

  return angles;

}

void IK::DoIKIndex() {

}

void IK::DoIKForward(Eigen::VectorXd angles) {
  //Eigen::VectorXd angles(5);
  //angles.fill(0.0f);
  /*angles(0) = -74.377;
  angles(1) = -99.6184;
  angles(2) = 0.03900;
  angles(3) = 0;*/
  Point3DVector points;
  points.push_back(marker[16].cast<double>());
  points.push_back(marker[17].cast<double>());
  points.push_back(marker[18].cast<double>());
  points.push_back(marker[4].cast<double>());
  //points.push_back(marker[6].cast<double>());
  //points.push_back(marker[5].cast<double>());

  HandFunctor functor;
  functor.Points = points;
  points = functor.ForwardStep(angles);

  for (int i = 0; i < points.size(); ++i) {
    std::cout << "Marker " << i << "at\t" << points[i].x() << " " << points[i].y() << " " << points[i].z() << "\n";
  }
  std::cout << "======================\n";
  Point3DVector res = functor.IndexFinger(angles);
  for (int i = 0; i < res.size(); ++i) {
    std::cout << "Marker " << i << "at\t" << res[i].x() << " " << res[i].y() << " " << res[i].z() << "\n";
  }
}

void IK::DoIK() {
  ProjectMarkers();

  Eigen::VectorXd angles(7);
  //set initial angles
  //for (int i = 0; angles.size(); ++i) {
  //  angles(i) = 0;
  //}
  angles.fill(0.0f);

  Point3DVector points;

  //root marker
  points.push_back(Eigen::Vector3d(0, 0, 0));

  //Eigen::Vector3d shift(2, 1, 0); //for index
  points.push_back(marker[4].cast<double>());
  points.push_back(marker[6].cast<double>());
  points.push_back(marker[5].cast<double>());

  //points.push_back(Eigen::Vector3d(0, 0, 0));
  HandFunctorNumericalDiff functor;
  functor.Points = points;

  Eigen::LevenbergMarquardt<HandFunctorNumericalDiff> lm(functor);

  Eigen::LevenbergMarquardtSpace::Status status = lm.minimize(angles);
  std::cout << "status: " << status << std::endl;
  std::cout << "x that minimizes the function: " << std::endl << angles << std::endl;

  //Save Results
  /*for (int i = 0; angles.size(); ++i) {
    angleResults[i] = angles(i);
  }*/
}

void IK::RootTransform() {

}

void IK::ConvertMmToCm() {
  for (int i = 0; i < 19; ++i) {
    marker[i] /= 10;
  }
}

//Palm is in xy plane
void IK::ProjectMarkers() {
  Eigen::Vector3f center = marker[16];
  for (int i = 0; i < 19; ++i) {
    marker[i] = marker[i] - center;
  }
  CalcRotationMatrix();
  
  for (int i = 0; i < 19; ++i)
  {
    //right order?
    marker[i] = rot * marker[i];
  }
}

void IK::CenterAround(int index) {
  Eigen::Vector3f center = marker[index];
  for (int i = 0; i < 19; ++i) {
    marker[i] = marker[i] - center;
  }

}

void IK::RotateMarkers() {
  for (int i = 0; i < 19; ++i) {
    marker[i] = rot * marker[i];
  }
}

void IK::FingerSpace(Eigen::Vector3f vec) {
  Eigen::Vector3f projected = Eigen::Vector3f(1, 0, 0); //in x
  Eigen::Vector3f v = vec.cross(projected);
  float c = vec.dot(projected);

  Eigen::Matrix3f skew;
  skew << 0, -v.z(), v.y(),
    v.z(), 0, -v.x(),
    -v.y(), v.x(), 0;
  rot = Eigen::Matrix3f::Identity() + skew + skew * skew * (1 / (1 + c));
}

void IK::CalcRotationMatrix() {
  //Check if really normalized
  Eigen::Vector3f normal = GetNormalBase().normalized();
  Eigen::Vector3f projected = Eigen::Vector3f(0, 0, -1);
  Eigen::Vector3f v = normal.cross(projected);
  //float s = v.norm();
  float c = normal.dot(projected);

  Eigen::Matrix3f skew;
  skew << 0, -v.z(), v.y(),
    v.z(), 0, -v.x(),
    -v.y(), v.x(), 0;

  rot = Eigen::Matrix3f::Identity() + skew + skew * skew * (1 / (1 + c));
}

Eigen::Vector3f IK::GetNormalBase()
{
  Eigen::Vector3f vec1 = marker[18] - marker[16];
  Eigen::Vector3f vec2 = marker[17] - marker[16];

  //cross product. check direction
  return vec1.cross(vec2);
}