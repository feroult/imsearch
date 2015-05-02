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

void detectEdges(Mat &src, Mat &dst)
{
    medianBlur( src, dst, 5 );
    cvtColor( dst, dst, CV_BGR2GRAY );
    GaussianBlur(dst, dst, Size(3, 3), 0);
    Canny( dst, dst, 30, 150 );
}

void detectLines(Mat &src, Mat &dst) {
    Mat mask = Mat::ones( src.size(), CV_8UC1 );
    Ptr<BinaryDescriptor> bd = BinaryDescriptor::createBinaryDescriptor();
    vector<KeyLine> lines;

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

            KeyLine fullKl = fullKeyLine(src, kl);

            Point pt1 = Point2f( fullKl.startPointX, fullKl.startPointY );
            Point pt2 = Point2f( fullKl.endPointX, fullKl.endPointY );

            line( dst, pt1, pt2, Scalar( 255, 255, 255 ), 1 );
        }
    }
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
    Mat image_edges = Mat::zeros( image.size(), CV_8UC1 );
    detectEdges(image, image_edges);

    Mat image_lines = Mat::zeros( image.size(), CV_8UC1 );
    detectLines(image_edges, image_lines);

    //Mat image_rects = Mat::zeros( image.size(), CV_8UC1 );
    Mat image_rects = image.clone();
    detectRects(image_lines, image_rects);

    /* show lines on image */
    imshow( "Lines", image_lines );
    waitKey();
}
