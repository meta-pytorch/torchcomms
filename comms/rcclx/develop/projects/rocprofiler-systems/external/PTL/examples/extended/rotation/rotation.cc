// MIT License
//
// Copyright (c) 2019 Jonathan R. Madsen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
//

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "common.hh"
#include "macros.hh"
#include "rotate_utils.hh"

using namespace cv;
using namespace std;

//============================================================================//

const char* source_label     = "Source image";
const char* forward_label    = "Forward Rotate";
const char* backward_label   = "Backward Rotate";
const char* difference_label = "Difference image";
const char* zoom_diff_label  = "Zoomed Difference";

//============================================================================//

void
plot(const char* label, const Mat& warp_mat)
{
    namedWindow(label, WINDOW_AUTOSIZE);
    imshow(label, warp_mat);
}

//============================================================================//

int
main(int argc, char** argv)
{
    string file  = "data/cameraman.tif";
    float  theta = -45.0f;
    float  scale = 1.0f;

    if(argc > 1)
        file = argv[1];
    if(argc > 2)
        theta = atof(argv[2]);
    if(argc > 3)
        scale = atof(argv[3]);

    std::vector<const char*> labels = { source_label, forward_label, backward_label,
                                        difference_label, zoom_diff_label };
    std::vector<Mat>         matrices;
    auto                     join = [&](std::vector<Mat>& src, Mat&& i) {
        src.push_back(std::move(i));
        return src;
    };
    auto run_man = cpu_run_manager();
    init_run_manager(run_man, std::thread::hardware_concurrency());
    auto                             tp = run_man->GetThreadPool();
    TaskGroup<std::vector<Mat>, Mat> tg(join, tp);
    int                              eInterp = GetOpenCVInterpolationMode();

    Mat  warp_src = imread(file);
    auto nx       = warp_src.rows;
    auto ny       = warp_src.cols;
    matrices.push_back(warp_src);

    tg.run(opencv_affine_transform, warp_src, -theta, nx, ny, eInterp, scale);
    matrices       = tg.join(matrices);
    Mat& warp_forw = matrices.back();

    tg.run(opencv_affine_transform, warp_forw, theta, nx, ny, eInterp, scale);
    matrices       = tg.join(matrices);
    Mat& warp_back = matrices.back();
    Mat  warp_diff = warp_src - warp_back;
    matrices.push_back(warp_diff);

    tg.run(opencv_affine_transform, warp_diff, 0.0f, nx, ny, eInterp,
           1.0f / cosf(theta * (M_PI / 180.f)));
    matrices = tg.join(matrices);

    assert(labels.size() == matrices.size());
    for(uint64_t i = 0; i < matrices.size(); ++i)
    {
        plot(labels[i], matrices[i]);
        waitKey(2000);
    }

    PrintEnv();

    return 0;
}
