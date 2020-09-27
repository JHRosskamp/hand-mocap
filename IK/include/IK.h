#pragma once
#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>
#include <Eigen/Sparse>
#include "Marker.h"

//3 marker per finger
struct finger {
  float m1, m2, m3;
};

typedef std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > Point3DVector;


// Generic functor
template<typename _Scalar, int NX = Eigen::Dynamic, int NY = Eigen::Dynamic>
struct Functor
{
  typedef _Scalar Scalar;
  enum {
    InputsAtCompileTime = NX,
    ValuesAtCompileTime = NY
  };
  typedef Eigen::Matrix<Scalar, InputsAtCompileTime, 1> InputType;
  typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, 1> ValueType;
  typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime> JacobianType;

  int m_inputs, m_values;

  Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
  Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

  int inputs() const { return m_inputs; }
  int values() const { return m_values; }

};

struct RootFunctor : Functor<double> {
  int operator()(const Eigen::VectorXd& x, Eigen::VectorXd& fvec) const
  {

  }
};

struct HandFunctor : Functor<double> {
  int operator()(const Eigen::VectorXd& x, Eigen::VectorXd& fvec) const
  {
    float l1 = 1;
    //int b = sind();
    //given angle x computes virtual marker position 

    //TODO set size for vector
    Point3DVector totalVector;
    //Position of root marker
    totalVector.push_back(Eigen::Vector3d(0, 0, 0));
  
    //Base
    Point3DVector root = RootIK(x);
    totalVector.insert(std::end(totalVector), std::begin(root), std::end(root));
    //Index
    Point3DVector index = IndexFinger(x);
    index = RotatedMarkerPosition(x, index);
    totalVector.insert(std::end(totalVector), std::begin(index), std::end(index));

    for (unsigned int i = 0; i < this->Points.size(); ++i) {
      fvec(i) = (totalVector[i] - this->Points[i]).squaredNorm();
    }

    return 0;
  }

  Point3DVector Points;
  //marker height
  double lmb = 0.2;

  double sind(double arg) const { return sin(arg * 3.14159265359 / 180); }
  double cosd(double arg) const { return cos(arg * 3.14159265359 / 180); }

  Point3DVector ForwardStep(const Eigen::VectorXd x) const {
    Point3DVector result;
    result.push_back(Eigen::Vector3d(0, 0, 0));
    Point3DVector root = RootIK(x);
    Point3DVector index = IndexFinger(x);
    index = RotatedMarkerPosition(x, index);
    result.insert(std::end(result), std::begin(root), std::end(root));
    result.insert(std::end(result), std::begin(index), std::end(index));
    return result;
  }

  Point3DVector RotatedMarkerPosition(const Eigen::VectorXd& x, const Point3DVector in) const {
    Point3DVector result;
    Eigen::Matrix3d R = RotationsMatrixXYZ(x(0),x(1),x(2));
    //Eigen::Matrix3d R = RotationsMatrixXYZ(-74.06, -99.52, 0.05);
    for (int i = 0; i < in.size(); ++i) {
      result.push_back(R * in[i]);
    }

    return result;
  }

  Point3DVector TranslateVector(Point3DVector vec, Eigen::Vector3d trans) const {
    for (int i = 0; i < vec.size(); ++i) {
      vec[i] += trans;
    }
    return vec;
  }

  //x is angle
  //l is length of finger
  Point3DVector LocalFingerPositionFromAngle(const Eigen::VectorXd& x, const Eigen::VectorXd& l, 
                                              const Eigen::VectorXd& mpos, const Eigen::VectorXd& mheight) const {
    Point3DVector result = SingleFingerIK(x, l, mpos, mheight);
    return result;
  }

  Point3DVector RootIK(const Eigen::VectorXd& x) const {

    Point3DVector result;
    Eigen::Matrix3d rot = RotationsMatrixXYZ(x(0), x(1), x(2));
    Eigen::Vector3d wrist = Eigen::Vector3d(-3.63348, 0.08594, -0.12949);
    Eigen::Vector3d base_thumb = Eigen::Vector3d(-1.75141, -4.257093, -0.646068);
    result.push_back(rot * wrist);
    result.push_back(rot * base_thumb);

    return result;
  }

  Point3DVector SingleFingerIK(const Eigen::VectorXd& x, const Eigen::VectorXd& l,
                                const Eigen::VectorXd& mpos, const Eigen::VectorXd& mheight) const {
    double lm = 0.2;
    //Unit vector for rest direction
    Eigen::Vector3d rest_dir = Eigen::Vector3d(1, 0, 0);
    //Unit vector for marker direction. Perpendicular to Rest direction
    Eigen::Vector3d marker_dir = Eigen::Vector3d(0, 0, 1);

    //x(0) is abduction. x(1-3) is flex angle
    Eigen::Matrix3d R1 = RotationMatrixYZ(0, x(0));// x(1));
    //Eigen::Matrix3d R2 = RotationMatrixYZ(0, x(2));
    //Eigen::Matrix3d R3 = RotationMatrixYZ(0, x(3));

    Point3DVector joint, rot_vec;
    rot_vec.push_back(rest_dir * 0);
    rot_vec.push_back(R1 * rest_dir * l(0));
    //rot_vec.push_back(R1 * R2 * rest_dir * l(1));
    //rot_vec.push_back(R1 * R2 * R3 * rest_dir * l(2));

    joint.push_back(rot_vec[0]);
    joint.push_back(rot_vec[1] + joint[0]);
    //joint.push_back(rot_vec[2] + joint[1]);
    //joint.push_back(rot_vec[3] + joint[2]);

    Point3DVector marker;
    Eigen::Vector3d trans;
    trans = 0.5 * rest_dir * 0;
    //marker.push_back(marker_dir * 0);
    trans = mpos(0) * rot_vec[1] + joint[0];
    marker.push_back(R1 * marker_dir * mheight(0) + trans);
    //trans = mpos(1) * rot_vec[2] + joint[1];
    //marker.push_back(R1 * R2 * marker_dir * mheight(1) + trans);
    //trans = mpos(2) * rot_vec[3] + joint[2];
    //trans += Eigen::Vector3d(0, 0.8, 0);
    //marker.push_back(R1 * R2 * R3 * marker_dir * mheight(2) + trans);

    return marker;
  }

  Point3DVector IndexFinger(const Eigen::VectorXd& angle) const {
    Eigen::VectorXd length(3), x(4), mpos(3), mheight(3);
    length << 4.5, 2.5, 2.7; //length
    mpos << 2.0 / 4.5, 1.64 / 2.5, 2.2 / 2.7;
    mheight << 2.3, 2.3, 2.3;
    x << angle(3), 0,0,0;// , angle(5), angle(6); //use the correct angles from all angles

    Eigen::Vector3d positionInHandModel = Eigen::Vector3d(4.92-2.0, -3.95, -1.13 - 2.3);
    //Eigen::Vector3d positionInHandModel = Eigen::Vector3d(2.5, -5, 0);

    Point3DVector localCoord = LocalFingerPositionFromAngle(x, length, mpos, mheight);
    localCoord = TranslateVector(localCoord, positionInHandModel);
    
    return localCoord;
    //Point3DVector tmp;
    //tmp.push_back(Eigen::Vector3d(5.3162, -1.00254, 3.4411));
    //return tmp;
  }

  Eigen::Matrix3d RotationMatrixYZ(double xy, double xz) const
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

  Eigen::Matrix3d RotationMatrixZ(double xy) const
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

  Eigen::Matrix3d RotationsMatrixXYZ(double alpha, double beta, double gamma) const {
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

  int inputs() const { return 5; } // There are two parameters of the model
  int values() const { return this->Points.size(); } // The number of observations

};

struct HandFunctorNumericalDiff : Eigen::NumericalDiff<HandFunctor> {};

class IK {
public:
  IK();
  //Copy data
  void setData(const std::vector<Marker>& data);
  void DoIK();
  Eigen::VectorXd DoIKRoot();
  void DoIKIndex();
  void DoIKTest();
  void DoIKForward(Eigen::VectorXd angles);
  void ConvertMmToCm();
  void FingerSpace(Eigen::Vector3f vec);
  void CenterAround(int index);
  void RotateMarkers();
  void ProjectMarkers();
  std::vector<Eigen::Vector3f> GetMarkerData();
  std::vector<float> GetAngleData() const {
    return angleResults;
  };
private:
  void RootTransform();
  void CalcRotationMatrix();
  Eigen::Vector3f GetNormalBase();

  Eigen::Matrix3f rot;
  std::vector<Eigen::Vector3f> marker;
  std::vector<float> angleResults;

};