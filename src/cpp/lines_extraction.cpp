/*///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2014, Biagio Montesano, all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/

#include <opencv2/line_descriptor.hpp>

#include "opencv2/core/utility.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

using namespace cv;
using namespace cv::line_descriptor;
using namespace std;

RNG rng(12345);

double ANGLE_THRESHOLD = 1.8;

static const char* keys =
{ "{@image_path | | Image path }" };

static void help()
{
  cout << "\nThis example shows the functionalities of lines extraction " << "furnished by BinaryDescriptor class\n"
       << "Please, run this sample using a command in the form\n" << "./example_line_descriptor_lines_extraction <path_to_input_image>" << endl;
}

void countors(Mat threshold_output)
{
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;

  /// Find contours
  findContours( threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

  /// Approximate contours to polygons + get bounding rects and circles
  vector<vector<Point> > contours_poly( contours.size() );
  vector<Rect> boundRect( contours.size() );
  vector<Point2f>center( contours.size() );
  vector<float>radius( contours.size() );

  for( int i = 0; i < contours.size(); i++ )
     { approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
       boundRect[i] = boundingRect( Mat(contours_poly[i]) );
       minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );
     }


  /// Draw polygonal contour + bonding rects + circles
  Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );
  for( int i = 0; i< contours.size(); i++ )
     {
       Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
       //drawContours( drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
       rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 1, 8, 0 );
       //circle( drawing, center[i], (int)radius[i], color, 2, 8, 0 );
     }

  /// Show in a window
  namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
  imshow( "Contours", drawing );
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

void fullLine(Mat img, Point a, Point b, Scalar color){

    double angle = Angle(a.x, a.y, b.x, b.y);

    if(angle == -1)
    {
        return;
    }

    if(angle == 0)
    {
        a.x = 2;
        b.x = img.cols - 2;
        b.y = a.y;
    }
    else
    {
        a.y = 2;
        b.y = img.rows - 2;
        b.x = a.x;
    }

    printf("angle: %f, a(%i, %i), b(%i, %i)\n", angle, a.x, a.y, b.x, b.y);

    line(img, a, b, color, 1);
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
  cv::Mat imageMat = imread( image_path, 1 );
  if( imageMat.data == NULL )
  {
    std::cout << "Error, image could not be loaded. Please, check its path" << std::endl;
    return -1;
  }

  /* change image */
  cv::Mat copy = imageMat.clone();
  medianBlur( copy, copy, 5 );

  //Canny( copy, copy, 30, 150, 5 );

  //medianBlur( imageMat, imageMat, 5 );
  //copy.convertTo(copy, -1, 2.2, 50);
  cvtColor( copy, copy, CV_BGR2GRAY );
  // GaussianBlur(copy, copy, Size(3, 3), 0);
  Canny( copy, copy, 30, 150 );

  //cvtColor( copy, copy, CV_BGR2GRAY );
  //threshold( copy, copy, 200, 255, THRESH_BINARY );
  //threshold( imageMat, imageMat, 100, 255, THRESH_BINARY );
  //imageMat.convertTo(imageMat, -1, 2.2, 50);

  /* create a random binary mask */
  cv::Mat mask = Mat::ones( copy.size(), CV_8UC1 );

  /* create a pointer to a BinaryDescriptor object with deafult parameters */
  Ptr<BinaryDescriptor> bd = BinaryDescriptor::createBinaryDescriptor();

  /* create a structure to store extracted lines */
  vector<KeyLine> lines;

  /* extract lines */
  bd->detect( copy, lines, mask );


  //cv::Mat output = imageMat.clone();
  cv::Mat output = Mat::zeros( imageMat.size(), CV_8UC1 );


  /* draw lines extracted from octave 0 */
  if( output.channels() == 1 )
    cvtColor( output, output, COLOR_GRAY2BGR );
  for ( size_t i = 0; i < lines.size(); i++ )
  {
    KeyLine kl = lines[i];
    if( kl.octave == 0)
    {
      /* get a random color */
      int R = ( rand() % (int) ( 255 + 1 ) );
      int G = ( rand() % (int) ( 255 + 1 ) );
      int B = ( rand() % (int) ( 255 + 1 ) );

      /* get extremes of line */
      Point pt1 = Point2f( kl.startPointX, kl.startPointY );
      Point pt2 = Point2f( kl.endPointX, kl.endPointY );

      /* draw line */
      //line( output, pt1, pt2, Scalar( R, G, B ), 2 );
      //fullLine( output, pt1, pt2, Scalar( R, G, B ));
      fullLine( output, pt1, pt2, Scalar( 255, 255, 255 ));
    }

  }

  //threshold( src_gray, dst, threshold_value, max_BINARY_value,threshold_type );
  cvtColor( output, output, CV_BGR2GRAY );

  //threshold( output, output, 254, 255, THRESH_BINARY );

  //countors(output);
  //cvtColor(output, output, CV_RGB2GRAY);

  std::vector<std::vector<Point> > contours;
  vector<Vec4i> hierarchy;


  //findContours( output, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE );
  //findContours( output, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE );
  findContours( output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );


  printf("size: %lu\n", contours.size());

  vector<vector<Point> > contours_poly( contours.size() );
  vector<Rect> boundRect( contours.size() );
  for( int i = 0; i < contours.size(); i++ )
      {
        if ( hierarchy[i][3] != -1 ) {
        approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
        boundRect[i] = boundingRect( Mat(contours_poly[i]) );
    }
      }

  //cv::Mat drawing = Mat::zeros( imageMat.size(), CV_8UC1 );
  cv::Mat drawing = imageMat.clone();
  for( int i = 0; i< contours.size(); i++ )
     {
       Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
       rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 1, 8, 0 );
     }

  /* show lines on image */
  imshow( "Lines", drawing );
  waitKey();
}
