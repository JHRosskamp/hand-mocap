#include <torch/torch.h>
#include <string>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <windows.h>
#include <chrono>
#include "MarkerFileInput.h"
#include "MarkerLabeling.h"
#include "ImagePCA.h"

namespace fs = std::filesystem;
using namespace std;

int main() {
    LoadLibrary("torch_cuda.dll");
    if (!torch::cuda::is_available()) {
      std::cout << "Cuda not available\n";
      return 1;
    }
    ImagePCA image;


    MarkerLabeling label("model-random.pt");// = MarkerLabeling("model-pca.pt");
    //label.
    
    size_t fileNumber = 0;
    size_t totalFileNumber = 1000;
    size_t totalFrameNumber = 0;

    std::vector<std::string> paths;
    paths.push_back("../TrainingData/User1/capture1");
    paths.push_back("../TrainingData/User1/capture2");
    paths.push_back("../TrainingData/User2/capture1");
    paths.push_back("../TrainingData/User2/capture2");
    paths.push_back("../TrainingData/User2/capture3");
    paths.push_back("../TrainingData/User2/capture4");
    paths.push_back("../TrainingData/User3/capture1");
    paths.push_back("../TrainingData/User3/capture2");
    paths.push_back("../TrainingData/User3/capture3");
    paths.push_back("../TrainingData/User3/capture4");
    paths.push_back("../TrainingData/User4/capture1");
    paths.push_back("../TrainingData/User4/capture2");
    paths.push_back("../TrainingData/User5/capture1");
    paths.push_back("../TrainingData/User5/capture2");
    auto start_time = std::chrono::high_resolution_clock::now();
    for (const auto& name : paths) {
        for (const auto& entry : fs::directory_iterator(name)) {
            if (fileNumber < totalFileNumber) {
                std::cout << entry.path() << std::endl;
                MarkerFileInput input(entry.path().string());
                input.readFile();
                auto frameNumber = input.numberOfFrames();
                size_t frames = 0;
                for (int j = 0; j < frameNumber; ++j) {
                    std::vector<Marker> marker = input.getNextFrame();
                    image.setInputMarker(marker);
                    inCNN data = image.createImage();
                    label.PredictAll(data);
                    ++totalFrameNumber;
                    ++frames;
                }
                ++fileNumber;
            }

        }
    }
    label.PrintAccuracy();
    auto end_time = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    std::cout << "Total time is = " << time << "s to run for " << totalFrameNumber << " frames " << std::endl;
	return 0;
}