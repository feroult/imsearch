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

// watermark detector

class WaterMarkDetector
{
    private:
        Mat src;
        vector<Mat> channels;
        vector<vector<ERStat> > regions;
        vector< vector<Vec2i> > region_groups;
        vector<Rect> groups_boxes;
        Ptr<ERFilter> er_filter1;
        Ptr<ERFilter> er_filter2;

    public:
        WaterMarkDetector(const char * filename);
        void applyClassifier();
        void detectGroups();
        void computeWaterMarkProbability();
        void release();
};

WaterMarkDetector::WaterMarkDetector(const char * filename)
{
    src = imread(filename);
}

void WaterMarkDetector::applyClassifier()
{
    computeNMChannels(src, channels);

    int cn = (int)channels.size();
    // Append negative channels to detect ER- (bright regions over dark background)
    for (int c = 0; c < cn-1; c++)
        channels.push_back(255-channels[c]);

    er_filter1 = createERFilterNM1(loadClassifierNM1("trained_classifierNM1.xml"),10,0.00015f,0.13f,0.8f,false,0.1f);
    er_filter2 = createERFilterNM2(loadClassifierNM2("trained_classifierNM2.xml"),0.5);

    // Apply the default cascade classifier to each independent channel (could be done in parallel)
    regions = vector<vector<ERStat> >(channels.size());
    for (int c=0; c<(int)channels.size(); c++)
    {
        er_filter1->run(channels[c], regions[c]);
        er_filter2->run(channels[c], regions[c]);
    }
}

void WaterMarkDetector::detectGroups()
{
    erGrouping(src, channels, regions, region_groups, groups_boxes, ERGROUPING_ORIENTATION_HORIZ, "./trained_classifier_erGrouping.xml", 0.5);
}

void WaterMarkDetector::computeWaterMarkProbability()
{

    Ptr<OCRHMMDecoder> ocr = create_ocr();

    string output;
    for (int i=0; i<(int)groups_boxes.size(); i++)
    {
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
}

void WaterMarkDetector::release()
{
    // memory clean-up
    er_filter1.release();
    er_filter2.release();
    regions.clear();
    if (!groups_boxes.empty())
    {
        groups_boxes.clear();
    }
}

// main function

int main(int argc, const char * argv[])
{
    if (argc < 2) show_help_and_exit(argv[0]);

    WaterMarkDetector wmd(argv[1]);
    wmd.applyClassifier();
    wmd.detectGroups();
    wmd.computeWaterMarkProbability();
    wmd.release();
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
