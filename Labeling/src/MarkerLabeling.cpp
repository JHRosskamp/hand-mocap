#include "MarkerLabeling.h"
#include <iostream>

MarkerLabeling::MarkerLabeling(std::string model) {
	
	outMarker.resize(19);
	bPredictedLabel.resize(19);
	//torch::load(net, "model-pca.pt");
	torch::load(net, "D:/VR-Phi/OptiTracking/bin/model-pca.pt");
	net->eval();
	net->to(torch::kCUDA);
	std::cout << "Success" << std::endl;
}

MarkerLabeling::MarkerLabeling() {
	outMarker.resize(19);
	bPredictedLabel.resize(19);
}

void MarkerLabeling::PredictAll(inCNN& data) {
	input = data;
	CallCNN();
	MatchingAll();
}

void MarkerLabeling::Predict(inCNN& data) {
	input = data;
	CallCNN();
	Matching();
}

std::vector<Marker> MarkerLabeling::GetMarker() {
	return outMarker;
}

std::vector<Marker> MarkerLabeling::GetMatched() {
	return input.normalized_marker;
}

void MarkerLabeling::SetLabels(std::vector<Marker>& data) {
	for (int i = 0; i < 19; ++i) {
		data[i].label = input.normalized_marker[i].label;
	}
}

void MarkerLabeling::CallCNN() {	
	//to tensor
	auto data = readImage(input.image).to(torch::kCUDA);
	//Call with input
	//std::cout << data.slice(0) << std::endl;
	auto out = net->forward(data).to(torch::kCUDA);
	//to vector
	//to cpu
	auto res = out.to(torch::kCPU);
	auto p = res.accessor<float, 3>();

	//Access first element
	auto p1 = p[0];
	for (int i = 0; i < p1.size(0); ++i) {
		auto p2 = p1[i];
		//std::cout << p2[0] << " " << p2[1] << " " << p2[2] << std::endl;
		Eigen::Vector3f tmp = Eigen::Vector3f(p2[0], p2[1], p2[2]);
		outMarker[i].pos = tmp;
		outMarker[i].label = marker_label(i);//input.normalized_marker[j].label;
	}
	//std::cout << p[0] << " " << p[1] << " " << p[2] << std::endl;
	/*int j = 0;
	for (int i = 0; i < 19; i += 3) {
		Eigen::Vector3f tmp = Eigen::Vector3f(p[i], p[i + 1], p[i + 2]);
		outMarker[j].pos = tmp;
		outMarker[j].label = marker_label(j);//input.normalized_marker[j].label; //Change here to array init;
		++j;
	}*/
}

void MarkerLabeling::PrintAccuracy() {
	std::cout << "Correct label = " << right << std::endl;
	std::cout << "Wrong label = " << wrong << std::endl;
}

void MarkerLabeling::MatchingAll() {
	std::vector<Eigen::Vector3f> in, pred;

	for (int i = 0; i < 19; ++i) {
		in.push_back(input.normalized_marker[i].pos);
		pred.push_back(outMarker[i].pos);
	}
	//Now matching.
	matching_result res = solve_matching_problem(in, pred);
	
	for (int i = 0; i < 19; ++i) {
		input.normalized_marker[i].label = res.labels[i];
	}
	//Match between input & outMarker

	for (int i = 0; i < 19; ++i) {
		if (res.labels[i] == marker_label(i)) {
			++right;
			bPredictedLabel[i] = true;
		}
		if (res.labels[i] != marker_label(i)) {
			++wrong;
			bPredictedLabel[i] = false;
		}
	}
	//std::cout << "Number Correct = " << right << " & Number False = " << wrong << std::endl;
}

void MarkerLabeling::Matching() {
	std::vector<Eigen::Vector3f> in, pred;
	std::vector<int> indexOut, indexIn;
	
	//index of markers without label
	for (int i = 0; i < 19; ++i) {
		if (input.normalized_marker[i].label == marker_label::no_label)
			indexIn.push_back(i);
	}

	//index of cnn out corresponding with markers without label
	for (int i = 0; i < 19; ++i) {
		bool labelUsed = false;
		for (int j = 0; j < 19; ++j) {
			if (marker_label(i) == input.normalized_marker[j].label) {
				labelUsed = true;
				continue;
			}
		}
		if (labelUsed == false) {
			indexOut.push_back(i);
		}
	}

	for (int i = 0; i < indexIn.size(); ++i) {
		in.push_back(input.normalized_marker[indexIn[i]].pos);
		pred.push_back(outMarker[indexOut[i]].pos);
	}
	
	//Now matching.
	matching_result res = solve_matching_problem(in, pred);
	for (int i = 0; i < indexIn.size(); ++i) {
		int index = static_cast<int>(res.labels[i]);
		input.normalized_marker[indexIn[i]].label = outMarker[indexOut[index]].label;
	}
}

torch::Tensor MarkerLabeling::readImage(depth_image& img) {
	const int imageSize = 52;
	Eigen::MatrixXf image = img.get_data();
	float mat[imageSize * imageSize];

	for (int rows = 0; rows < imageSize; ++rows) {
		for (int cols = 0; cols < imageSize; ++cols) {
			mat[cols + rows * imageSize] = image(rows, cols);
		}
	}
	torch::Tensor imageTensor = torch::from_blob(mat, { imageSize,imageSize,1,1 });
	//std::cout << imageTensor.sizes() << std::endl;
	imageTensor = imageTensor.permute({ 2, 3, 0, 1 });
	//std::cout << imageTensor.sizes() << std::endl;
	//imageTensor.view({ 1, 1, imageSize, imageSize });
	return imageTensor.clone();
}