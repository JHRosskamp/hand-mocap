#include "IK.h"


double sind(double arg) { return sin(arg * 3.14159265359 / 180); }
double cosd(double arg) { return cos(arg * 3.14159265359 / 180); }

Eigen::Matrix3d RotationMatrixZ(double xy)
{
  Eigen::Matrix3d retval;
  retval(0) = cosd(xy);
  retval(1) = -sind(xy);
  retval(2) = 0;
  retval(3) = sind(xy);
  retval(4) = cosd(xy);
  retval(5) = 0;
  retval(6) = 0;
  retval(7) = 0;
  retval(8) = 1;
  return retval;
}

Eigen::Matrix3d RotationMatrixYZ(double xy, double xz)
{
  Eigen::Matrix3d retval;
  retval(0) = cosd(xy) * cosd(xz);
  retval(1) = -sind(xy);
  retval(2) = cosd(xy) * sind(xz);
  retval(3) = cosd(xz) * sind(xy);
  retval(4) = cosd(xy);
  retval(5) = sind(xy) * sind(xz);
  retval(6) = -sind(xz);
  retval(7) = 0;
  retval(8) = cosd(xz);
  return retval;
}

Eigen::Matrix3d RotationsMatrixXYZ(double alpha, double beta, double gamma) {
  Eigen::Matrix3d retval;
  retval(0) = cosd(alpha) * cosd(gamma) - sind(alpha) * cosd(beta) * sind(gamma);
  retval(1) = -cosd(alpha) * sind(gamma) - sind(alpha) * cosd(beta) * cosd(gamma);
  retval(2) = sind(alpha) * sind(beta);
  retval(3) = sind(alpha) * cosd(gamma) + cosd(alpha) * cosd(beta) * sind(gamma);
  retval(4) = -sind(alpha) * sind(gamma) + cosd(alpha) * cosd(beta) * cosd(gamma);
  retval(5) = -cosd(alpha) * sind(beta);
  retval(6) = sind(beta) * sind(gamma);
  retval(7) = sind(beta) * cosd(gamma);
  retval(8) = cosd(beta);
  return retval;
}

Point3DVector TranslateVector(Point3DVector vec, Eigen::Vector3d trans) {
  for (int i = 0; i < vec.size(); ++i) {
    vec[i] += trans;
  }
  return vec;
}

Point3DVector SingleFingerIK(const Eigen::VectorXd& x, const Eigen::VectorXd& l,
  const Eigen::VectorXd& mpos, const Eigen::VectorXd& mheight, const Eigen::VectorXd& d) {
  double lm = 0.2;
  //Unit vector for rest direction
  Eigen::Vector3d rest_dir = Eigen::Vector3d(1, 0, 0);
  //Unit vector for marker direction. Perpendicular to Rest direction
  Eigen::Vector3d marker_dir = Eigen::Vector3d(0, 0, 1);

  //x(0) is abduction. x(1-3) is flex angle
  Eigen::Matrix3d R1 = RotationMatrixYZ(x(0), x(1));
  Eigen::Matrix3d R2 = RotationMatrixYZ(0, x(2));
  Eigen::Matrix3d R3 = RotationMatrixYZ(0, x(3));

  Point3DVector joint, rot_vec;
  rot_vec.push_back(rest_dir * 0);
  rot_vec.push_back(R1 * rest_dir * l(0));
  rot_vec.push_back(R1 * R2 * rest_dir * l(1));
  rot_vec.push_back(R1 * R2 * R3 * rest_dir * l(2));

  joint.push_back(rot_vec[0]);
  joint.push_back(rot_vec[1] + joint[0]);
  joint.push_back(rot_vec[2] + joint[1]);
  joint.push_back(rot_vec[3] + joint[2]);

  Point3DVector marker;
  Eigen::Vector3d trans;
  trans = 0.5 * rest_dir * 0;
  //mpos changes depending on angle
  d(0) * sind(x(1));
  d(1) * sind(x(2)) + d(1) * sind(x(3));
  d(2) * sind(x(3));
  //marker.push_back(marker_dir * 0);
  trans = mpos(0) * rot_vec[1] + joint[0];
  marker.push_back(R1 * marker_dir * mheight(0) + trans);
  trans = mpos(1) * rot_vec[2] + joint[1];
  marker.push_back(R1 * R2 * marker_dir * mheight(1) + trans);
  trans = mpos(2) * rot_vec[3] + joint[2];
  marker.push_back(R1 * R2 * R3 * marker_dir * mheight(2) + trans);

  return marker;
}

Point3DVector LocalFingerPositionFromAngle(const Eigen::VectorXd& x, const Eigen::VectorXd& l,
  const Eigen::VectorXd& mpos, const Eigen::VectorXd& mheight, const Eigen::VectorXd& d) {
  Point3DVector result = SingleFingerIK(x, l, mpos, mheight,d);
  return result;
}

Point3DVector IndexFinger(const Eigen::VectorXd& angle) {
  Eigen::VectorXd length(3), x(4), mpos(3), mheight(3), d(3);
  length << 4.5, 2.5, 2.7; //length
  mpos << 2.0 / 4.5, 1.64 / 2.5, 2.2 / 2.7;
  mheight << 2.3, 2.3, 2.3; //should stay constant
  x << angle(1), angle(2), angle(3), angle(4); //use the correct angles from all angles
  d << 1, 0.75, 0.75; //half of finger thickness

  //Eigen::Vector3d positionInHandModel = Eigen::Vector3d(2.5, -5, 0);
  Eigen::Vector3d positionInHandModel = Eigen::Vector3d(4.9 - 2.0, -3.95, -1.13 - 2.3);
  Point3DVector localCoord = LocalFingerPositionFromAngle(x, length, mpos, mheight, d);
  localCoord = TranslateVector(localCoord, positionInHandModel);

  return localCoord;
}

Point3DVector RotatedMarkerPosition(const Eigen::VectorXd x, const Point3DVector in) {
  Point3DVector result;
  Eigen::Matrix3d R = RotationsMatrixXYZ(x(0),x(1),x(2));
  for (int i = 0; i < in.size(); ++i) {
    result.push_back(R * in[i]);
  }

  return result;
}

int main() {
  //IK ik;
  //ik.DoIKTest();
  Eigen::VectorXd angle(5), rot(3);
  angle(0) = 0;
  angle(1) = 0;
  angle(2) = 0;
  angle(3) = 0;
  angle(4) = 0;
  rot(0) = 0;
  rot(1) = 0;
  rot(2) = 0;
  HandFunctor test;
  Point3DVector points;
  points = IndexFinger(angle);
  Eigen::Vector3d center = points[0];
  points = RotatedMarkerPosition(rot, points);
  for (int i = 0; i < points.size(); ++i) {
    //points[i] -= center;
    std::cout << "Marker " << i << " is at " << points[i].x() << "\t" << points[i].y() << "\t" << points[i].z() << std::endl;
  }
  //std::cout << "dist " << (points[0] - points[1]).norm() << std::endl;
  //std::cout << "dist2 " << (points[1] - points[2]).norm() << std::endl;
  return 0;
}