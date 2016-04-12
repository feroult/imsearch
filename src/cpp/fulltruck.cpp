#include <opencv2/line_descriptor.hpp>

#include "opencv2/core/utility.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

using namespace cv;
using namespace line_descriptor;
using namespace std;

RNG rng(12345);

double ANGLE_THRESHOLD = 1.8;

struct limit_type {
    int l1;
    int l2;
};

static const char* keys =
{ "{@image_path | | Image path }" };

static void help()
{
  cout << "./fulltruck <path_to_input_image>" << endl;
}

double Angle(int x0, int y0, int x1, int y1){
    double angle = abs(atan2(y1 - y0, x1 - x0) * 180.0 / CV_PI);

    if(angle < ANGLE_THRESHOLD)
    {
        return 0;
    }

    if(abs(angle - 90) < ANGLE_THRESHOLD)
    {
        return 90;
    }

    if(abs(angle - 180) < ANGLE_THRESHOLD)
    {
        return 0;
    }

    if(abs(angle - 270) < ANGLE_THRESHOLD)
    {
        return 90;
    }

    if(abs(angle - 360) < ANGLE_THRESHOLD)
    {
        return 0;
    }

    return -1;

}

bool isValidLine(KeyLine kl)
{
    double angle = Angle(kl.startPointX, kl.startPointY, kl.endPointX, kl.endPointY);
    return angle != -1;
}

KeyLine fullKeyLine(Mat image, KeyLine kl)
{

    KeyLine fullKl;

    double angle = Angle(kl.startPointX, kl.startPointY, kl.endPointX, kl.endPointY);

    fullKl.startPointX = kl.startPointX;
    fullKl.startPointY = kl.startPointY;
    fullKl.endPointX = kl.endPointX;
    fullKl.endPointY = kl.endPointY;
    fullKl.angle = angle;

    if(angle == 0)
    {
        fullKl.startPointX = 2;
        fullKl.endPointX = image.cols - 2;
        fullKl.endPointY = fullKl.startPointY;
    }
    else
    {
        fullKl.startPointY = 2;
        fullKl.endPointY = image.rows - 2;
        fullKl.endPointX = fullKl.startPointX;
    }

    return fullKl;
}

void prepareImage(Mat &src, Mat &dst)
{
    src.copyTo(dst);
    cvtColor( dst, dst, CV_BGR2GRAY );
    GaussianBlur(dst, dst, Size(5, 5), 0);
    Canny( dst, dst, 0, 100*3, 3 );
}

vector<KeyLine> detectLines(Mat &src, Mat &dst) {
    Mat mask = Mat::ones( src.size(), CV_8UC1 );
    Ptr<BinaryDescriptor> bd = BinaryDescriptor::createBinaryDescriptor();
    vector<KeyLine> lines;

    vector<KeyLine> fullLines;

    bd->detect( src, lines, mask );

    if( dst.channels() == 1 )
    {
        cvtColor( dst, dst, COLOR_GRAY2BGR );
    }

    for ( size_t i = 0; i < lines.size(); i++ )
    {
        KeyLine kl = lines[i];
        if( kl.octave == 0)
        {
            if(!isValidLine(kl))
            {
                continue;
            }

            //KeyLine fullKl = fullKeyLine(src, kl);
            KeyLine fullKl = kl;

            Point pt1 = Point2f( fullKl.startPointX, fullKl.startPointY );
            Point pt2 = Point2f( fullKl.endPointX, fullKl.endPointY );

            line( dst, pt1, pt2, Scalar( 255, 255, 255 ), 1 );

            fullLines.push_back(fullKl);
        }
    }

    return fullLines;
}

void detectRects(Mat &_src, Mat &dst)
{
    Mat src = _src.clone();
    cvtColor( src, src, CV_BGR2GRAY );

    std::vector<std::vector<Point> > contours;
    vector<Vec4i> hierarchy;

    findContours( src, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

    vector<vector<Point> > contours_poly( contours.size() );
    vector<Rect> boundRect( contours.size() );
    for( int i = 0; i < contours.size(); i++ )
    {
        if ( hierarchy[i][3] != -1 )
        {
            approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
            boundRect[i] = boundingRect( Mat(contours_poly[i]) );
        }
    }

    for( int i = 0; i< contours.size(); i++ )
    {
        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        rectangle( dst, boundRect[i].tl(), boundRect[i].br(), color, 1, 8, 0 );
    }
}

// kill me
void detectCircles(Mat &_src, Mat &dst)
{
    Mat src = _src.clone();
    src.copyTo(dst);

    cvtColor( src, src, CV_BGR2GRAY );

    //GaussianBlur( src, src, Size(9, 9), 2, 2 );

    vector<Vec3f> circles;

    // threshold( src, src, 80, 255, 2 );
    // threshold( src, src, 70, 255, 3 );

    HoughCircles( src, circles, CV_HOUGH_GRADIENT, 1, src.rows/8, 200, 90, 0, 0 );

    printf("circles: %lu\n", circles.size());

    for( size_t i = 0; i < circles.size(); i++ )
    {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        circle( dst, center, 3, Scalar(255,255,255), -1, 8, 0 );
        circle( dst, center, radius, Scalar(255,255,255), 3, 8, 0 );
    }
}

int BIN_GROUP = 10;
double MEAN_THRESHOLD = 1;

limit_type calcLimits(Mat _src, int size, bool horizontal)
{
    Mat src = _src.clone();

    cvtColor( src, src, CV_BGR2GRAY );
    GaussianBlur(src, src, Size(3, 3), 0);
    Canny( src, src, 0, 100*3, 3 );
    threshold(src, src, 254, 255, CV_THRESH_BINARY);

    int chunks = size/BIN_GROUP;

    int total = 0;
    int* histogram = new int[chunks];
    for(int i = 0; i < chunks; i++)
    {
        int sum = 0;
        for(int j = 0; j < BIN_GROUP; j++)
        {
            if(horizontal)
            {
                sum = countNonZero(src.col(i*BIN_GROUP+j));
            }
            else
            {
                sum = countNonZero(src.row(i*BIN_GROUP+j));
            }
        }

        histogram[i] = sum;
        total += sum;
        printf("sum: %i - %i\n", i, sum);
    }

    float mean = (float)total/chunks;

    limit_type limits;

    for(int i = 0; i < chunks; i++)
    {
        if(histogram[i] > mean * MEAN_THRESHOLD)
        {
            limits.l1 = i * BIN_GROUP + BIN_GROUP/2;
            printf("l1: %i - %i\n", limits.l1, i);
            break;
        }
    }

    for(int i = chunks-1; i >= 0; i--)
    {
        if(histogram[i] > mean * MEAN_THRESHOLD)
        {
            limits.l2 = i * BIN_GROUP + BIN_GROUP/2;;
            printf("l2: %i - %i\n", limits.l2, i);
            break;
        }
    }

    return limits;
}

void calcPixelByColumn(Mat _src, Mat &dst)
{
    limit_type hlimits = calcLimits(_src, _src.cols, true);

    line( dst, Point2f(hlimits.l1, 0), Point2f(hlimits.l1, dst.rows), Scalar( 255, 255, 255 ), 5 );
    line( dst, Point2f(hlimits.l2, 0), Point2f(hlimits.l2, dst.rows), Scalar( 255, 255, 255 ), 5 );

    limit_type vlimits = calcLimits(_src, _src.rows, false);

    line( dst, Point2f(0, vlimits.l1), Point2f(dst.cols, vlimits.l1), Scalar( 255, 255, 255 ), 5 );
    line( dst, Point2f(0, vlimits.l2), Point2f(dst.cols, vlimits.l2), Scalar( 255, 255, 255 ), 5 );
}

int* calcPixelByColumnX(Mat _src, Mat &dst)
{
    Mat src = _src.clone();

    cvtColor( src, src, CV_BGR2GRAY );
    GaussianBlur(src, src, Size(3, 3), 0);
    Canny( src, src, 0, 100*3, 3 );
    threshold(src, src, 254, 255, CV_THRESH_BINARY);

    int width = src.cols/BIN_GROUP;

    int total = 0;
    int* histogram = new int[width];
    for(int i = 0; i < width; i++)
    {
        int col_total = 0;
        for(int j = 0; j < BIN_GROUP; j++)
        {
            col_total = countNonZero(src.col(i*BIN_GROUP+j));
        }

        histogram[i] = col_total;
        total += col_total;
        printf("col: %i - %i\n", i, col_total);
    }

    float mean = (float)total/width;

    int left, right;

    for(int i = 0; i < width; i++)
    {
        if(histogram[i] > mean * MEAN_THRESHOLD)
        {
            left = i * BIN_GROUP + BIN_GROUP/2;
            printf("left: %i - %i\n", left, i);
            break;
        }
    }

    for(int i = width-1; i >= 0; i--)
    {
        if(histogram[i] > mean * MEAN_THRESHOLD)
        {
            right = i * BIN_GROUP + BIN_GROUP/2;;
            printf("right: %i - %i\n", right, i);
            break;
        }
    }

    line( dst, Point2f(left, 0), Point2f(left, dst.rows), Scalar( 255, 255, 255 ), 5 );
    line( dst, Point2f(right, 0), Point2f(right, dst.rows), Scalar( 255, 255, 255 ), 5 );

    printf("mean - total: %i - %f\n", total, mean);

    return histogram;
}

int main( int argc, char** argv )
{
    /* get parameters from comand line */
    CommandLineParser parser( argc, argv, keys );
    String image_path = parser.get<String>( 0 );

    if( image_path.empty() )
    {
        help();
        return -1;
    }

    /* load image */
    Mat image = imread( image_path, 1 );
    if( image.data == NULL )
    {
        std::cout << "Error, image could not be loaded. Please, check its path" << std::endl;
        return -1;
    }

    /* change image */
    Mat image_copy = Mat::zeros( image.size(), CV_8UC1 );
    prepareImage(image, image_copy);

    //Mat image_lines = Mat::zeros( image.size(), CV_8UC1 );
    Mat image_lines = image.clone();
    vector<KeyLine> lines = detectLines(image_copy, image_lines);
    printf("lines: %lu\n", lines.size());

    //Mat image_rects = Mat::zeros( image.size(), CV_8UC1 );
    Mat image_rects = image.clone();
    detectRects(image_lines, image_rects);

    // Mat image_circles = Mat::zeros( image.size(), CV_8UC1 );
    // detectCircles(image, image_circles);

    //Mat image_pixels = image.clone();
    Mat image_pixels = image_lines.clone();
    calcPixelByColumn(image, image_pixels);

    imshow( "Lines", image_pixels );
    waitKey();
}
