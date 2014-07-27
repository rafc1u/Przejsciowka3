/*

//#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <fstream>
#include <iostream>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include <string>
#include <opencv2/nonfree/features2d.hpp>
#include <math.h>


	int median(std::vector<int> &v){
		size_t n = v.size() / 2;
		nth_element(v.begin(), v.begin()+n, v.end());
		return v[n];
	}

int main(){



	cv:: Mat zdjl, zdjr, imagel, imager, outImgl, outImgr, matchess, matchess2;
	int wartSlider = 18000;
	std::vector<cv::KeyPoint> keypointsl, keypointsr;;

	cv::SurfDescriptorExtractor surfDesc;
	cv::Mat descriptorsl, descriptorsr;

	//zdjl = cv::imread("C:\\left.jpg", CV_LOAD_IMAGE_COLOR );
	//zdjr = cv::imread("C:\\right.jpg", CV_LOAD_IMAGE_COLOR );

	//zdjl = cv::imread("C:\\Users\\kkk\\Documents\\Visual Studio 2010\\Projects\\HelloGUIWorld2\\Win32\\Debug\\lewa00.jpg", CV_LOAD_IMAGE_COLOR );
	//zdjr = cv::imread("C:\\Users\\kkk\\Documents\\Visual Studio 2010\\Projects\\HelloGUIWorld2\\Win32\\Debug\\prawa00.jpg", CV_LOAD_IMAGE_COLOR );

	zdjl = cv::imread("C:\\Users\\kkk\\Desktop\\leftp2.jpg", CV_LOAD_IMAGE_COLOR );
	zdjr = cv::imread("C:\\Users\\kkk\\Desktop\\rightp2.jpg", CV_LOAD_IMAGE_COLOR );


	while(cv::waitKey(10) != 27){
		imshow("left", zdjl);
		imshow("right", zdjr);



		zdjl.copyTo(imagel);
		zdjr.copyTo(imager);

		cv::SurfFeatureDetector surf(wartSlider);
		surf.detect(imagel, keypointsl);
		surf.detect(imager, keypointsr);

		

		drawKeypoints(imagel, keypointsl, outImgl, cv::Scalar(255,255,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		drawKeypoints(imager, keypointsr, outImgr, cv::Scalar(255,255,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

		imshow("Detected left", outImgl);
		imshow("Detected right", outImgr);


		surfDesc.compute(imagel, keypointsl, descriptorsl);
		surfDesc.compute(imager, keypointsr, descriptorsr);

		cv::BruteForceMatcher<cv::L2<float>> matcher;
		std::vector<cv::DMatch> matches;
		matcher.match(descriptorsl, descriptorsr, matches);
		
		
		

		std::vector<int> pointIndexesLeft;
		std::vector<int> pointIndexesRight;
		
		for (std::vector<cv::DMatch>::const_iterator it= matches.begin(); it!= matches.end(); ++it) {

				 // Get the indexes of the selected matched keypoints
				 pointIndexesLeft.push_back(it->queryIdx);
				 pointIndexesRight.push_back(it->trainIdx);
		}

		//iteracja po wektorach - obliczanie odleglosci euklidesowej miedzy x,y pointIndexLeft a x,y pointIndexesRight. NAstepnie mediana i weryfikacja czy otrzymane pary maja mediane odl euk. +-1
		// z wektora keypointa o adresie wskazywanym przez iterator left or right pobieramy pt - czyli x oraz y
		std::vector<int>::const_iterator itl = pointIndexesLeft.begin();
		std::vector<int>::const_iterator itr = pointIndexesRight.begin();
		//wektory odl_eu - powinno byc ze to wektor wspolczynnikow a nachylen prostych
		//wektor old_y - 
		std::vector<double> odl_eu;

		//kopie pointindexes
		std::vector<int> pointIndexesLeft2 (pointIndexesLeft);
		std::vector<int> pointIndexesRight2 (pointIndexesRight);

		//kopie keypoint's
		std::vector<cv::KeyPoint> keypointsl2 (keypointsl);
		std::vector<cv::KeyPoint> keypointsr2 (keypointsr);


		//kopia matches
		std::vector<cv::DMatch> matches2(matches);
		//pomocnicza zmienna zeby nastepowalo przesuniecie gdy usuwamy z listy wektorow
			int pp = 0;

		
		//filtrowanie
		for (int i = 0; i < pointIndexesLeft.size(); i++){

			double wynik;
			cv::Point left;
			cv::Point right;
			int abc = *itl;
			left = keypointsl.at(*itl).pt;
			right = keypointsr.at(*itr).pt;
			wynik = tan( (double) (double(left.y-right.y)/(double(left.x-right.x)) ) );
			odl_eu.push_back( wynik );
			
			
			//wyrzucenie POWINNA BYC MEDIANA
			if (wynik != 0 ){
				matches2.erase(matches2.begin()+i-pp);
				pointIndexesLeft2.erase(pointIndexesLeft2.begin() + i-pp);
				pointIndexesRight2.erase(pointIndexesRight2.begin() + i-pp);
				//czy napewno?
				//keypointsl2.erase(keypointsl2.begin() + i);
				//keypointsr2.erase(keypointsr2.begin() + i);
				pp++;
			}
			 itr++;
			 itl++;
		}
		//pointIndexes - przypisanie poprawnych
			pointIndexesLeft = pointIndexesLeft2;
			pointIndexesRight = pointIndexesRight2;

		// Convert keypoints into Point2f
			std::vector<cv::Point2f> selPointsLeft, selPointsRight;
			
			cv::KeyPoint::convert(keypointsl,selPointsLeft,pointIndexesLeft);
			cv::KeyPoint::convert(keypointsr,selPointsRight,pointIndexesRight);

			cv::Mat fundemental= cv::findFundamentalMat(
            cv::Mat(selPointsLeft), // points in first image
            cv::Mat(selPointsRight), // points in second image
            CV_FM_RANSAC);       // 8-point method

		
			
			// rysowanie DOPASOWAN
			cv::drawMatches(imagel,keypointsl,imager, keypointsr, matches2, matchess);
			imshow("Matches without correct function", matchess);

			// rysowanie DOPASOWAN
			//cv::drawMatches(imagel,keypointsl,imager, keypointsr, matches2, matchess);
			//imshow("Matches with correct function", matchess2);

			cv::Mat(H1);
			cv::Mat(H2);

			cv::stereoRectifyUncalibrated(selPointsLeft, selPointsRight, fundemental, imagel.size(), H1,H2);

			cv::Mat  w, u, vt;
			cv::SVD::compute(fundemental, w, u, vt);	

			//Epipole to ostatnia kolumna macierzy V, tutaj jest po transpozycji wiec ew moze byc zle wziety- wiec tu szukac bledu!!!!!!
			cv::Mat ep_left;
			vt.col(2).copyTo(ep_left);

			cv::Mat ep_right = u.col(2);

			cv::Mat ep_left2 (3,3, CV_64FC1);
			cv::Mat ep_right2 (3,3, CV_64FC1);

			ep_left2 = (0, -ep_left.at<double>(2,0), ep_left.at<double>(1,0), 
						ep_left.at<double>(2), 0, -ep_left.at<double>(0), 
						-ep_left.at<double>(1), ep_left.at<double>(0), 0 );


			ep_left2.at<double>(0,1) = -ep_left.at<double>(2,0);
			ep_left2.at<double>(0,2) = ep_left.at<double>(1,0);
			ep_left2.at<double>(1,0) = ep_left.at<double>(2);
			ep_left2.at<double>(1,2) = -ep_left.at<double>(0);
			ep_left2.at<double>(2,0) = -ep_left.at<double>(1);
			ep_left2.at<double>(2,1) = ep_left.at<double>(0);

			//std::cout << "-ep_left.at<double>(2,0): \n" << -ep_left.at<double>(2,0) << std::endl;
			//std::cout << "ep_left.at<double>(1,0): \n" << ep_left.at<double>(1,0) << std::endl;


			ep_right2 = (0, -ep_right.at<double>(2), ep_right.at<double>(1), 
						ep_right.at<double>(2), 0, -ep_right.at<double>(0), 
						-ep_right.at<double>(1), ep_right.at<double>(0), 0 );

			ep_right2.at<double>(0,1) = -ep_right.at<double>(2,0);
			ep_right2.at<double>(0,2) = ep_right.at<double>(1,0);
			ep_right2.at<double>(1,0) = ep_right.at<double>(2);
			ep_right2.at<double>(1,2) = -ep_right.at<double>(0);
			ep_right2.at<double>(2,0) = -ep_right.at<double>(1);
			ep_right2.at<double>(2,1) = ep_right.at<double>(0);




			cv::Mat P_lp (3,3, CV_64FC1);
			cv::Mat P_lp2 (3,4, CV_64FC1);
			cv::Mat P_rp (3,3, CV_64FC1);
			cv::Mat P_rp2 (3,4, CV_64FC1);
			


			P_lp = (ep_left2 * fundemental);
			P_rp = (ep_right2 * fundemental);

			//std::cout << "Macierz fundemental: \n" << fundemental << std::endl;
			//std::cout << "Macierz ep_left: \n" << ep_left << std::endl;
			//std::cout << "Macierz ep_left2: \n" << ep_left2 << std::endl;
			//std::cout << "Macierz ep_right2: \n" << ep_right2 << std::endl;

			for (int i = 0; i < 3; i++) {
				P_lp.col(i).copyTo(P_lp2.col(i));
				P_rp.col(i).copyTo(P_rp2.col(i));
			}

			ep_left.copyTo(P_lp2.col(3));		
			ep_right.copyTo(P_rp2.col(3));	

			double m[3][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}};
			cv::Mat Pr(3,4, CV_64FC1, m);
			//Pr = (1, 0, 0, 0, 
			//	  0, 1, 0, 0,
			//	  0, 0, 1, 0,
			//	  0, 0, 0, 0);

			cv:: Mat poTriangulacji(1, keypointsl.size(), CV_64FC4);
			cv:: Mat dst(1, keypointsl.size(), CV_64FC3);
			cv::triangulatePoints(Pr, P_rp2, selPointsLeft, selPointsRight, poTriangulacji);

			//std::cout << "Macierz Pr: \n" << Pr << std::endl;
			//std::cout << "Macierz P_rp2: \n" << P_rp2 << std::endl;
			//std::cout << "Macierz selPointsLeft: \n" << selPointsLeft << std::endl;
			//std::cout << "Macierz selPointsRight: \n" << selPointsRight << std::endl;
			std::cout << "Macierz fundemental: \n" << fundemental << std::endl;
			
			
			//std::cout << "Macierz potriangulacji: \n" << poTriangulacji<< std::endl;

		//	cv::convertPointsFromHomogeneous(poTriangulacji,dst);
	}

	return 0;
}

*/