/*
 * textdetection.cpp
 *
 * A demo program of the Extremal Region Filter algorithm described in
 * Neumann L., Matas J.: Real-Time Scene Text Localization and Recognition, CVPR 2012
 *
 * Created on: Sep 23, 2013
 *     Author: Lluis Gomez i Bigorda <lgomez AT cvc.uab.es>
 */

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
void groups_draw(Mat &src, vector<Rect> &groups);
void er_show(vector<Mat> &channels, vector<vector<ERStat> > &regions);
void er_draw(vector<Mat> &channels, vector<vector<ERStat> > &regions, vector<Vec2i> group, Mat& segmentation);
bool isRepetitive(const string& s);

int main(int argc, const char * argv[])
{
    cout << endl << argv[0] << endl << endl;
    cout << "Demo program of the Extremal Region Filter algorithm described in " << endl;
    cout << "Neumann L., Matas J.: Real-Time Scene Text Localization and Recognition, CVPR 2012" << endl << endl;

    if (argc < 2) show_help_and_exit(argv[0]);

    Mat src = imread(argv[1]);

    // cvtColor( src, src, CV_BGR2GRAY );
    // GaussianBlur(src, src, Size(3, 3), 0);
    // Canny( src, src, 30, 150 );


    // Extract channels to be processed individually
    vector<Mat> channels;
    computeNMChannels(src, channels);

    int cn = (int)channels.size();
    // Append negative channels to detect ER- (bright regions over dark background)
    for (int c = 0; c < cn-1; c++)
        channels.push_back(255-channels[c]);

    // Create ERFilter objects with the 1st and 2nd stage default classifiers
    // Ptr<ERFilter> er_filter1 = createERFilterNM1(loadClassifierNM1("trained_classifierNM1.xml"),16,0.00015f,0.13f,0.2f,true,0.1f);
    // Ptr<ERFilter> er_filter2 = createERFilterNM2(loadClassifierNM2("trained_classifierNM2.xml"),0.5);

    // cb	– Callback with the classifier. Default classifier can be implicitly load with function loadClassifierNM1(), e.g. from file in samples/cpp/trained_classifierNM1.xml
    // thresholdDelta	– Threshold step in subsequent thresholds when extracting the component tree
    // minArea	– The minimum area (% of image size) allowed for retreived ER’s
    // maxArea	– The maximum area (% of image size) allowed for retreived ER’s
    // minProbability	– The minimum probability P(er|character) allowed for retreived ER’s
    // nonMaxSuppression	– Whenever non-maximum suppression is done over the branch probabilities
    // minProbabilityDiff	– The minimum probability difference between local maxima and local minima ERs

    Ptr<ERFilter> er_filter1 = createERFilterNM1(loadClassifierNM1("trained_classifierNM1.xml"),10,0.00015f,0.13f,0.8f,false,0.1f);
    Ptr<ERFilter> er_filter2 = createERFilterNM2(loadClassifierNM2("trained_classifierNM2.xml"),0.5);

    vector<vector<ERStat> > regions(channels.size());
    // Apply the default cascade classifier to each independent channel (could be done in parallel)
    cout << "Extracting Class Specific Extremal Regions from " << (int)channels.size() << " channels ..." << endl;
    cout << "    (...) this may take a while (...)" << endl << endl;
    for (int c=0; c<(int)channels.size(); c++)
    {
        er_filter1->run(channels[c], regions[c]);
        er_filter2->run(channels[c], regions[c]);
    }

    // Detect character groups
    cout << "Grouping extracted ERs ... ";
    vector< vector<Vec2i> > region_groups;
    vector<Rect> groups_boxes;
    // erGrouping(src, channels, regions, region_groups, groups_boxes, ERGROUPING_ORIENTATION_HORIZ);
    erGrouping(src, channels, regions, region_groups, groups_boxes, ERGROUPING_ORIENTATION_HORIZ, "./trained_classifier_erGrouping.xml", 0.5);

    // OCR

    double t_r = (double)getTickCount();
    // Ptr<OCRTesseract> ocr = OCRTesseract::create();


    Mat transition_p;
    string filename = "OCRHMM_transitions_table.xml";
    FileStorage fs(filename, FileStorage::READ);
    fs["transition_probabilities"] >> transition_p;
    fs.release();
    Mat emission_p = Mat::eye(62,62,CV_64FC1);
    string voc = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

    Ptr<OCRHMMDecoder> ocr = OCRHMMDecoder::create(loadOCRHMMClassifierNM("OCRHMM_knn_model_data.xml.gz"),
                                                   voc, transition_p, emission_p);

    cout << "TIME_OCR_INITIALIZATION = " << ((double)getTickCount() - t_r)*1000/getTickFrequency() << endl;
    string output;


    cout << "HERE 0";

    Mat out_img;
    Mat out_img_detection;
    Mat out_img_segmentation = Mat::zeros(src.rows+2, src.cols+2, CV_8UC1);
    src.copyTo(out_img);
    src.copyTo(out_img_detection);
    float scale_img  = 600.f/src.rows;
    float scale_font = (float)(2-scale_img)/1.4f;
    vector<string> words_detection;

    t_r = (double)getTickCount();

    for (int i=0; i<(int)groups_boxes.size(); i++)
    {

        cout << "HERE 1";
        rectangle(out_img_detection, groups_boxes[i].tl(), groups_boxes[i].br(), Scalar(0,255,255), 3);

        Mat group_img = Mat::zeros(src.rows+2, src.cols+2, CV_8UC1);
        er_draw(channels, regions, region_groups[i], group_img);
        Mat group_segmentation;
        group_img.copyTo(group_segmentation);
        //image(groups_boxes[i]).copyTo(group_img);
        group_img(groups_boxes[i]).copyTo(group_img);
        copyMakeBorder(group_img,group_img,15,15,15,15,BORDER_CONSTANT,Scalar(0));


        // imshow("grouping",group_img);
        // waitKey();


        vector<Rect>   boxes;
        vector<string> words;
        vector<float>  confidences;

        cout << "HERE";

        ocr->run(group_img, output, &boxes, &words, &confidences, OCR_LEVEL_WORD);

        output.erase(remove(output.begin(), output.end(), '\n'), output.end());
        cout << "OCR output = \"" << output << "\" lenght = " << output.size() << endl;
        if (output.size() < 3)
            continue;

        for (int j=0; j<(int)boxes.size(); j++)
        {
            boxes[j].x += groups_boxes[i].x-15;
            boxes[j].y += groups_boxes[i].y-15;

            cout << "  word = " << words[j] << "\t confidence = " << confidences[j] << endl;
            if ((words[j].size() < 2) || (confidences[j] < 51) ||
                    ((words[j].size()==2) && (words[j][0] == words[j][1])) ||
                    ((words[j].size()< 4) && (confidences[j] < 60)) ||
                    isRepetitive(words[j]))
                continue;
            words_detection.push_back(words[j]);
            rectangle(out_img, boxes[j].tl(), boxes[j].br(), Scalar(255,0,255),3);
            Size word_size = getTextSize(words[j], FONT_HERSHEY_SIMPLEX, (double)scale_font, (int)(3*scale_font), NULL);
            rectangle(out_img, boxes[j].tl()-Point(3,word_size.height+3), boxes[j].tl()+Point(word_size.width,0), Scalar(255,0,255),-1);
            putText(out_img, words[j], boxes[j].tl()-Point(1,1), FONT_HERSHEY_SIMPLEX, scale_font, Scalar(255,255,255),(int)(3*scale_font));
            out_img_segmentation = out_img_segmentation | group_segmentation;
        }

    }

    cout << "TIME_OCR = " << ((double)getTickCount() - t_r)*1000/getTickFrequency() << endl;

    // OCR



    // draw groups
    groups_draw(src, groups_boxes);
    imshow("grouping",src);

    cout << "Done!" << endl << endl;
    cout << "Press 'space' to show the extracted Extremal Regions, any other key to exit." << endl << endl;
    if ((waitKey()&0xff) == ' ')
        er_show(channels,regions);

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

void groups_draw(Mat &src, vector<Rect> &groups)
{
    for (int i=(int)groups.size()-1; i>=0; i--)
    {
        if (src.type() == CV_8UC3)
            rectangle(src,groups.at(i).tl(),groups.at(i).br(),Scalar( 0, 255, 255 ), 3, 8 );
        else
            rectangle(src,groups.at(i).tl(),groups.at(i).br(),Scalar( 255 ), 3, 8 );
    }
}

void er_show(vector<Mat> &channels, vector<vector<ERStat> > &regions)
{
    for (int c=0; c<(int)channels.size(); c++)
    {
        Mat dst = Mat::zeros(channels[0].rows+2,channels[0].cols+2,CV_8UC1);
        for (int r=0; r<(int)regions[c].size(); r++)
        {
            ERStat er = regions[c][r];
            if (er.parent != NULL) // deprecate the root region
            {
                int newMaskVal = 255;
                int flags = 4 + (newMaskVal << 8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;
                floodFill(channels[c],dst,Point(er.pixel%channels[c].cols,er.pixel/channels[c].cols),
                          Scalar(255),0,Scalar(er.level),Scalar(0),flags);
            }
        }
        char buff[10]; char *buff_ptr = buff;
        sprintf(buff, "channel %d", c);
        imshow(buff_ptr, dst);
    }
    waitKey(-1);
}

bool isRepetitive(const string& s)
{
    int count = 0;
    for (int i=0; i<(int)s.size(); i++)
    {
        if ((s[i] == 'i') ||
                (s[i] == 'l') ||
                (s[i] == 'I'))
            count++;
    }
    if (count > ((int)s.size()+1)/2)
    {
        return true;
    }
    return false;
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
