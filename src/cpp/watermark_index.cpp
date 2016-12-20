#include  "opencv2/text.hpp"
#include  "opencv2/highgui.hpp"
#include  "opencv2/imgproc.hpp"

#include  <vector>
#include  <iostream>
#include  <iomanip>

using namespace std;
using namespace cv;
using namespace cv::text;

void show_help_and_exit(const char *cmd);
void er_draw(vector<Mat> &channels, vector<vector<ERStat> > &regions, vector<Vec2i> group, Mat& segmentation);
Ptr<OCRHMMDecoder> create_ocr();

int main(int argc, const char * argv[])
{
    if (argc < 2) show_help_and_exit(argv[0]);

    Mat src = imread(argv[1]);

    // GaussianBlur(src, src, Size(1, 1), 0);

    // Extract channels to be processed individually
    vector<Mat> channels;
    computeNMChannels(src, channels);

    int cn = (int)channels.size();
    // Append negative channels to detect ER- (bright regions over dark background)
    for (int c = 0; c < cn-1; c++)
        channels.push_back(255-channels[c]);

    Ptr<ERFilter> er_filter1 = createERFilterNM1(loadClassifierNM1("trained_classifierNM1.xml"),10,0.00015f,0.13f,0.8f,false,0.1f);
    Ptr<ERFilter> er_filter2 = createERFilterNM2(loadClassifierNM2("trained_classifierNM2.xml"),0.5);

    // Apply the default cascade classifier to each independent channel (could be done in parallel)
    vector<vector<ERStat> > regions(channels.size());
    for (int c=0; c<(int)channels.size(); c++)
    {
        er_filter1->run(channels[c], regions[c]);
        er_filter2->run(channels[c], regions[c]);
    }

    // Detect character groups
    vector< vector<Vec2i> > region_groups;
    vector<Rect> groups_boxes;
    erGrouping(src, channels, regions, region_groups, groups_boxes, ERGROUPING_ORIENTATION_HORIZ, "./trained_classifier_erGrouping.xml", 0.5);

    // OCR
    Ptr<OCRHMMDecoder> ocr = create_ocr();

    string output;

    Mat out_img;
    Mat out_img_detection;
    Mat out_img_segmentation = Mat::zeros(src.rows+2, src.cols+2, CV_8UC1);
    src.copyTo(out_img);
    src.copyTo(out_img_detection);
    float scale_img  = 600.f/src.rows;
    float scale_font = (float)(2-scale_img)/1.4f;
    vector<string> words_detection;
    for (int i=0; i<(int)groups_boxes.size(); i++)
    {
        rectangle(out_img_detection, groups_boxes[i].tl(), groups_boxes[i].br(), Scalar(0,255,255), 3);
        cout << "BOX: " << groups_boxes[i].width << " - x: " << groups_boxes[i].x << " y: " << groups_boxes[i].y << " - src: " << src.cols << endl;

        Mat group_img = Mat::zeros(src.rows+2, src.cols+2, CV_8UC1);
        er_draw(channels, regions, region_groups[i], group_img);
        Mat group_segmentation;
        group_img.copyTo(group_segmentation);
        group_img(groups_boxes[i]).copyTo(group_img);
        copyMakeBorder(group_img, group_img, 15, 15, 15, 15, BORDER_CONSTANT, Scalar(0));

        vector<Rect>   boxes;
        vector<string> words;
        vector<float>  confidences;
        ocr->run(group_img, output, &boxes, &words, &confidences, OCR_LEVEL_WORD);

        output.erase(remove(output.begin(), output.end(), '\n'), output.end());
        if (output.size() < 1)
            continue;

        for (int j=0; j<(int)boxes.size(); j++)
        {
            boxes[j].x += groups_boxes[i].x-15;
            boxes[j].y += groups_boxes[i].y-15;

            cout << "box: " << boxes[j].width << endl;
            cout << "  word = " << words[j] << "\t confidence = " << confidences[j] << endl;
        }

    }

    // OCR

    // memory clean-up
    er_filter1.release();
    er_filter2.release();
    regions.clear();
    if (!groups_boxes.empty())
    {
        groups_boxes.clear();
    }
}

// helper functions

void show_help_and_exit(const char *cmd)
{
    cout << "    Usage: " << cmd << " <input_image> " << endl;
    cout << "    Default classifier files (trained_classifierNM*.xml) must be in current directory" << endl << endl;
    exit(-1);
}

void er_draw(vector<Mat> &channels, vector<vector<ERStat> > &regions, vector<Vec2i> group, Mat& segmentation)
{
    for (int r=0; r<(int)group.size(); r++)
    {
        ERStat er = regions[group[r][0]][group[r][1]];
        if (er.parent != NULL) // deprecate the root region
        {
            int newMaskVal = 255;
            int flags = 4 + (newMaskVal << 8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;
            floodFill(channels[group[r][0]],segmentation,Point(er.pixel%channels[group[r][0]].cols,er.pixel/channels[group[r][0]].cols),
                      Scalar(255),0,Scalar(er.level),Scalar(0),flags);
        }
    }
}

Ptr<OCRHMMDecoder> create_ocr()
{
    Mat transition_p;
    string filename = "OCRHMM_transitions_table.xml";
    FileStorage fs(filename, FileStorage::READ);
    fs["transition_probabilities"] >> transition_p;
    fs.release();
    Mat emission_p = Mat::eye(62,62,CV_64FC1);
    string voc = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

    Ptr<OCRHMMDecoder> ocr = OCRHMMDecoder::create(loadOCRHMMClassifierNM("OCRHMM_knn_model_data.xml.gz"),
                                                   voc, transition_p, emission_p);
    return ocr;
}
