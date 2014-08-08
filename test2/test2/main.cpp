//#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <fstream>
#include <iostream>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include <string>
#include <opencv2/nonfree/features2d.hpp>
#include <math.h>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#define PI 3.14159265
#define ka PI/180

template <class type>
type mediana(std::vector<type> &v);

enum nachylenie{
			
				brak = 0,
				dodatnie = 1,
				ujemne = 2,
			};

void Ocena(std::vector<cv::DMatch> matches, std::vector<cv::KeyPoint> keypointsl, std::vector<cv::KeyPoint> keypointsr, std::vector<double> *odleglosci, std::vector<nachylenie> *vnachyl){

	std::vector<cv::DMatch>::iterator itDM = matches.begin();
	
	
	std::cout << "\nParametry kolejnych par: " << std::endl;
	for (int i = 0; i < matches.size(); i++){
		double dlug;
		cv::Point left = keypointsl.at(itDM->queryIdx).pt;
		cv::Point right = keypointsr.at(itDM->trainIdx).pt;
		dlug = sqrt(double(double(left.x-right.x)*double(left.x-right.x)) + double(double(left.y-right.y)*double(left.y-right.y))  );
		//*odleglosci.push_back(dlug);
		odleglosci->push_back(dlug);
		if (left.y < right.y){
			vnachyl->push_back(dodatnie);
			std::cout << i << ": Dlugosc: " << dlug << ", wspolczynnik a =  dodatnie" << std::endl;
		}
		else if (left.y == right.y){
			vnachyl->push_back(brak);
			std::cout << i << ": Dlugosc: " << dlug << ", wspolczynnik a =  0" << std::endl;
		}
		else{
			vnachyl->push_back(ujemne);
			std::cout << i << ": Dlugosc: " << dlug << ", wspolczynnik a =  ujemne" << std::endl;
		}
		itDM++;
	}

}

void Ocena(std::vector<cv::DMatch> matches, std::vector<cv::KeyPoint> keypointsl, std::vector<cv::KeyPoint> keypointsr){

	std::vector<cv::DMatch>::iterator itDM = matches.begin();
	std::vector<double> odleglosci;
	std::vector<nachylenie> vnachyl;
	
	std::cout << "Parametry kolejnych par: " << std::endl;
	for (int i = 0; i < matches.size(); i++){
		double dlug;
		cv::Point left = keypointsl.at(itDM->queryIdx).pt;
		cv::Point right = keypointsr.at(itDM->trainIdx).pt;
		dlug = sqrt(double(double(left.x-right.x)*double(left.x-right.x)) + double(double(left.y-right.y)*double(left.y-right.y))  );
		//*odleglosci.push_back(dlug);
		odleglosci.push_back(dlug);
		if (left.y < right.y){
			vnachyl.push_back(dodatnie);
			std::cout << i << ": Dlugosc: " << dlug << ", wspolczynnik a =  dodatnie" << std::endl;
		}
		else if (left.y == right.y){
			vnachyl.push_back(brak);
			std::cout << i << ": Dlugosc: " << dlug << ", wspolczynnik a =  0" << std::endl;
		}
		else{
			vnachyl.push_back(ujemne);
			std::cout << i << ": Dlugosc: " << dlug << ", wspolczynnik a =  ujemne" << std::endl;
		}
		itDM++;
	}

}

std::vector<cv::DMatch> Filtrowanie (std::vector<cv::KeyPoint> keypointsl, std::vector<cv::KeyPoint> keypointsr,
				  std::vector<cv::DMatch> *matches, cv::Mat imagel, cv::Mat imager){

				//std::vector<int>::const_iterator itl = pointIndexesLeft.begin();
				//std::vector<int>::const_iterator itr = pointIndexesRight.begin();
				//Zamiast pointIndexes left i right uzywamy itDM
				std::vector<cv::DMatch>::iterator itDM;
				std::vector<double> odleglosci, odleglosci2;
				std::vector<nachylenie> vnachyl;
				cv::Mat imageMatchesPrzed, imageMatchesPo;
				std::vector<cv::DMatch> matchesPrawidlowe;

				Ocena(*matches, keypointsl, keypointsr, &odleglosci, &vnachyl);
				itDM = matches->begin();
				std::vector<double> tmp_odleglosci(odleglosci);
				std::vector<nachylenie> tmp_vnachyl(vnachyl);
				//Wyliczenie mediany odleglosci laczacych punkty charakterystyczne oraz wsp馧czynniki nachylenia tych prostych
				double mOdleglosc = mediana(tmp_odleglosci);
				std::cout << "Mediana odleglosci punktow char. = " << mOdleglosc << std::endl;
				nachylenie mNachylenie = mediana(tmp_vnachyl);
				if (mNachylenie == dodatnie){
					std::cout << "Mediana nachylen jest dodatnia" << std::endl;
				}
				else if(mNachylenie == ujemne){
					std::cout << "Mediana nachylen jest ujemna" << std::endl;
				}
				else{
					std::cout << "Mediana nachylen jest rowna 0" << std::endl;

				}
				
				
				// rysowanie DOPASOWAN PRZED FILTRACJA
				cv::drawMatches(imagel,keypointsl,imager, keypointsr, *matches, imageMatchesPrzed);
				cv::imshow("Before Filtration", imageMatchesPrzed);

				//Proces odfiltrowania
				//itl = pointIndexesLeft.begin();
				//itr = pointIndexesRight.begin();
				

				std::vector<double>::iterator itOdl = odleglosci.begin();
				std::vector<nachylenie>::iterator itNach = vnachyl.begin();

				for ( int i = 0; i < odleglosci.size(); i ++ ){

					// dla przypadku idealnego przesuniecia
					//if (*itOdl == mOdleglosc && *itNach == mNachylenie){
					//dla przypadku bardziej ogolnego niz wyzej
					if (*itOdl > ( mOdleglosc - 15.0) && (*itOdl < ( mOdleglosc + 15.0)) && *itNach == mNachylenie){

						cv::DMatch dmatch;
						dmatch.queryIdx = itDM->queryIdx;
						dmatch.trainIdx = itDM->trainIdx;
						matchesPrawidlowe.push_back(dmatch);
						
					}

					itOdl++;
					itNach++;
					itDM++;
				}
				// rysowanie DOPASOWAN Po FILTRACJI
				cv::drawMatches(imagel,keypointsl,imager, keypointsr, matchesPrawidlowe, imageMatchesPo);
				cv::imshow("After Filtration", imageMatchesPo);

				Ocena(matchesPrawidlowe,keypointsl, keypointsr, &odleglosci2, &vnachyl);
				return matchesPrawidlowe;
}


	double median(std::vector<double> &v){
		size_t n = v.size() / 2;
		nth_element(v.begin(), v.begin()+n, v.end());
		return v[n];
	}

	template <class type>
	type mediana(std::vector<type> &v){
		size_t n = v.size() / 2;
		nth_element(v.begin(), v.begin()+n, v.end());
		return v[n];
	}





int main(){


	cv:: Mat zdjl, zdjr, imagel, imager, outImgl, outImgr, matchess, matchess2;
	int wartSlider = 15000; //18000 uzywac
	std::vector<cv::KeyPoint> keypointsl, keypointsr;;

	cv::SurfDescriptorExtractor surfDesc;
	cv::Mat descriptorsl, descriptorsr;

	//zdjl = cv::imread("C:\\left.jpg", CV_LOAD_IMAGE_COLOR );
	//zdjr = cv::imread("C:\\right.jpg", CV_LOAD_IMAGE_COLOR );

	//zdjl = cv::imread("C:\\Users\\kkk\\Documents\\Visual Studio 2010\\Projects\\HelloGUIWorld2\\Win32\\Debug\\lewa00.jpg", CV_LOAD_IMAGE_COLOR );
	//zdjr = cv::imread("C:\\Users\\kkk\\Documents\\Visual Studio 2010\\Projects\\HelloGUIWorld2\\Win32\\Debug\\prawa00.jpg", CV_LOAD_IMAGE_COLOR );

	zdjl = cv::imread("C:\\Users\\Rafa許\Desktop\\mgr_obrazki\\leftp2.jpg", CV_LOAD_IMAGE_COLOR );
	zdjr = cv::imread("C:\\Users\\Rafa許\Desktop\\mgr_obrazki\\rightp2.jpg", CV_LOAD_IMAGE_COLOR );

	zdjl = cv::imread("C:\\Users\\Rafa許\Desktop\\mgr_obrazki\\dol.jpg", CV_LOAD_IMAGE_COLOR ); //
	zdjr = cv::imread("C:\\Users\\Rafa許\Desktop\\mgr_obrazki\\dol3.jpg", CV_LOAD_IMAGE_COLOR );// przesuniecie, obrot zmiana skali
	
	

	float tabRot[2][2] = {{cos(ka*45), sin(ka*45)},{-sin(ka*45), cos(ka*45)}};
	cv::Mat matRot(2,2, CV_32F, tabRot);

	float tabTr[2][2] = {{1,0},{0,1}};
	cv::Mat matTr(2,2, CV_32F, tabTr);

	cv::Mat matZlozenia = matRot*matTr;
	//std::cout << "Mac zlozenia:\n" << matZlozenia << std::endl;

	const int wys = 300;
	const int szer = 400;
	//zamalowanie obrazu na bialo
	float tabZdj[wys][szer];
	float tabZdj2[wys][szer];
	for (int i = 0; i < wys; i++){
		for (int j = 0; j < szer; j++){
			tabZdj[i][j] = 255;
			tabZdj2[i][j] = 255;
		}
	}

	
	for (int i = 0; i < (wys - 30); i += 20){
		for (int j = 0; j < (szer - 30); j += 20){
			tabZdj[i][j] = 0;
			int x = i;
			int y = j;
			double xp = matZlozenia.at<float>(0,0)*x+ matZlozenia.at<float>(0,1)*y;
			double yp = matZlozenia.at<float>(1,0)*x+ matZlozenia.at<float>(1,1)*y;
			if (   (  ((int)xp >= 0) && (int)xp < szer   ) && ((int)yp >=0 && (int)yp< wys) ){
				tabZdj2[(int)xp][(int)yp] = 0;	
			}
		}
	}

	cv::Mat matZdjl(300,400, CV_32F, tabZdj);

	cv::Mat matZdjr(300,400, CV_32F, tabZdj2);
	


	float transformMatrix[3][3] = {{cos(ka*45),sin(ka*45),-150},{-sin(ka*45),cos(ka*45),100},{0,0,1}};
	cv::Mat M(3,3, CV_32F, transformMatrix);
	//std::cout <<M<<std::endl;
	//cv::warpPerspective(zdjl,zdjr, M, zdjl.size());
	
	while(cv::waitKey(10) != 27){
		//imshow("left", zdjl);
		//imshow("right", zdjr);
	


		zdjl.copyTo(imagel);
		zdjr.copyTo(imager);

		cv::SurfFeatureDetector surf(wartSlider);
		surf.detect(imagel, keypointsl);
		surf.detect(imager, keypointsr);

		

		drawKeypoints(imagel, keypointsl, outImgl, cv::Scalar(255,255,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		drawKeypoints(imager, keypointsr, outImgr, cv::Scalar(255,255,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);


		surfDesc.compute(imagel, keypointsl, descriptorsl);
		surfDesc.compute(imager, keypointsr, descriptorsr);

		cv::BruteForceMatcher<cv::L2<float>> matcher;
		std::vector<cv::DMatch> matches, matchesPrawidlowe, knnMatchesPrawidlowe;
		std::vector<std::vector<cv::DMatch>> knnMatches;

		matcher.knnMatch(descriptorsl, descriptorsr, knnMatches, 2);
		matcher.match(descriptorsl, descriptorsr, matches);
		
		// knn warunek na distance

		for (std::vector<std::vector<cv::DMatch>>::iterator it = knnMatches.begin(); it != knnMatches.end(); it++){
			//for ( std::vector<cv::DMatch>::iterator it2 = it->begin(); it2 != it->end() ; it2++){
			//	std::cout << it2->distance << "\n" << std::endl;
			//}
			std::vector<cv::DMatch>::iterator it2 = it->begin();
			cv::DMatch tmpDMatch;
			double dist1 = -1;
			double dist2 = -1;
			dist1 = it2->distance;
			it2++;
			dist2 = it2->distance;
			it2--;

			if(dist1 < 0.65*dist2){
				tmpDMatch.queryIdx = it2->queryIdx;
				tmpDMatch.trainIdx = it2->trainIdx;
				knnMatchesPrawidlowe.push_back(tmpDMatch);
			}else{
			//	std::cout<<"zle dopas"<<std::endl;
			}
			
			
		}
		cv:: Mat knnimgmatches;
		cv:: drawMatches(imagel,keypointsl,imager, keypointsr, knnMatchesPrawidlowe, knnimgmatches);
		imshow("Matches from distance condition ", knnimgmatches);



		std::vector<int> pointIndexesLeft;
		std::vector<int> pointIndexesRight;

		//Filtrowanie
		//matchesPrawidlowe = Filtrowanie(keypointsl, keypointsr, &matches, imagel, imager);
		//2matchesPrawidlowe = matches;
		std::cout << "Poza funkcja filtracji:" << std::endl;
		//Ocena(matchesPrawidlowe,keypointsl, keypointsr);
		//pointIndexes - przypisanie poprawnych
		for (std::vector<cv::DMatch>::iterator it = knnMatchesPrawidlowe.begin(); it < knnMatchesPrawidlowe.end(); it++){
			pointIndexesLeft.push_back(it->queryIdx );
			pointIndexesRight.push_back(it->trainIdx );
		}

		// Convert keypoints into Point2f
			std::vector<cv::Point2f> selPointsLeft, selPointsRight;
			
			cv::KeyPoint::convert(keypointsl,selPointsLeft,pointIndexesLeft);
			cv::KeyPoint::convert(keypointsr,selPointsRight,pointIndexesRight);
			
			
			//Wyliczenie macierzy fundemental
			cv::Mat fundemental= cv::findFundamentalMat(
            cv::Mat(selPointsLeft), // points in first image
            cv::Mat(selPointsRight), // points in second image
            CV_FM_RANSAC);       // 8-point method
			std::cout << "\nfundemental\n" << fundemental << std::endl;
			cv::Mat l_prim;
			//l_prim = fundemental*
			

			//wyzncznik
			double wyznFund = cv::determinant(fundemental);
			std::cout << " Wyznacnzik fundamanetal matrix = " << wyznFund << std::endl;


			//sprawdzenie wg wzoru 11.2 ze strony 279 z MVG
			std::cout << "Checking the correctness (should be 0) " << std::endl;
			for (int i = 0; i < selPointsRight.size(); i ++){

				cv::Point2f xy = selPointsLeft.at(i);
				cv::Point2f xyp = selPointsRight.at(i);
				double x = xy.x;
				double y = xy.y;
				double xp = xyp.x;
				double yp = xyp.y;
				double f11 = fundemental.at<double>(0,0);
				double f12 = fundemental.at<double>(0,1);
				double f13 = fundemental.at<double>(0,2);
				double f21 = fundemental.at<double>(1,0);
				double f22 = fundemental.at<double>(1,1);
				double f23 = fundemental.at<double>(1,2);
				double f31 = fundemental.at<double>(2,0);
				double f32 = fundemental.at<double>(2,1);
				double f33 = fundemental.at<double>(2,2);

				double kontrola = xp*x*f11+xp*y*f12+xp*f13+yp*x*f21+yp*y*f22+yp*f23+x*f31+y*f32+f33;
				std::cout << "Kontrola: " << kontrola << std::endl;
			}
			//cv::SVD svd(fundemental,cv::SVD::MODIFY_A);
			cv::SVD svd(fundemental);
			std::cout << std::endl<< "SVD_U " << svd.u << std::endl;
			std::cout << std::endl<< "SVD_VT " << svd.vt << std::endl;
			std::cout << std::endl<< "SVD_W " << svd.w << std::endl;
			cv::Mat svd_u = svd.u;
			cv::Mat svd_vt = svd.vt;
			cv::Mat svd_w = svd.w;
			cv::Matx33d W(0,-1,0,
						1,0,0,
						0,0,1);

			cv::Matx33d Wt(0,1,0,
						-1,0,0,
						 0,0,1);

			cv::Mat_<double> R = svd_u * cv::Mat(W).t() * svd_vt; //or svd_u * Mat(W) * svd_vt; 
			cv::Mat_<double> R2 = svd_u * cv::Mat(W) * svd_vt; //or svd_u * Mat(W) * svd_vt; 
			cv::Mat_<double> t = svd_u.col(2); //or -svd_u.col(2)

			std::cout << std::endl<< "R " << R << std::endl;
			std::cout << std::endl<< "R2 " << R2 << std::endl;
			std::cout << "\nt " << t << std::endl;
			std::cout << "\nMacierz u z ktorej bierze sie t \n" << svd_u << std::endl;

			cv::Mat H1 = cv::findHomography(selPointsLeft, selPointsRight, 0);
			std::cout << "\n H1:\n" << H1 << std::endl;
			cv::Mat(H2);
			std::cout << "\n H2:\n" << H2 << std::endl;

			//cv::stereoRectifyUncalibrated(selPointsLeft, selPointsRight, fundemental, imagel.size(), H1,H2);

			cv::Mat  w, u, vt, v;
			cv::SVD::compute(fundemental, w, u, vt);	
			
			cv::transpose(vt,v);

			//Epipole to ostatnia kolumna macierzy V, tutaj jest po transpozycji wiec ew moze byc zle wziety- wiec tu szukac bledu!!!!!!

			cv::Mat ep_left, ep_leftO; //ep_leftO - opcja druga po tranzspoyzcja (poszukiwania bledu)
			vt.col(2).copyTo(ep_left);
			v.col(2).copyTo(ep_leftO); //szukanie bledu	

			cv::Mat ep_right = u.col(2);
			cv::Mat ep_rightO = u.col(2); // szukanie bledu

			cv::Mat ep_left2 (3,3, CV_64FC1);
			cv::Mat ep_right2 (3,3, CV_64FC1);
			cv::Mat ep_left2O (3,3, CV_64FC1); // szukanie bledu
			cv::Mat ep_right2O (3,3, CV_64FC1); // szukanie bledu

			ep_left2 = (0, -ep_left.at<double>(2,0), ep_left.at<double>(1,0), 
						ep_left.at<double>(2), 0, -ep_left.at<double>(0), 
						-ep_left.at<double>(1), ep_left.at<double>(0), 0 );

			ep_left2O = (0, -ep_leftO.at<double>(2,0), ep_leftO.at<double>(1,0), //SZUKANIE BLEDU
						ep_leftO.at<double>(2), 0, -ep_leftO.at<double>(0), 
						-ep_leftO.at<double>(1), ep_leftO.at<double>(0), 0 );


			ep_left2.at<double>(0,1) = -ep_left.at<double>(2,0);
			ep_left2.at<double>(0,2) = ep_left.at<double>(1,0);
			ep_left2.at<double>(1,0) = ep_left.at<double>(2);
			ep_left2.at<double>(1,2) = -ep_left.at<double>(0);
			ep_left2.at<double>(2,0) = -ep_left.at<double>(1);
			ep_left2.at<double>(2,1) = ep_left.at<double>(0);


			ep_left2O.at<double>(0,1) = -ep_leftO.at<double>(2,0); //SZUKANIE BLEDU
			ep_left2O.at<double>(0,2) = ep_leftO.at<double>(1,0); //SZUKANIE BLEDU
			ep_left2O.at<double>(1,0) = ep_leftO.at<double>(2); //SZUKANIE BLEDU
			ep_left2O.at<double>(1,2) = -ep_leftO.at<double>(0); //SZUKANIE BLEDU
			ep_left2O.at<double>(2,0) = -ep_leftO.at<double>(1); //SZUKANIE BLEDU
			ep_left2O.at<double>(2,1) = ep_leftO.at<double>(0); //SZUKANIE BLEDU

			

			ep_right2 = (0, -ep_right.at<double>(2), ep_right.at<double>(1), 
						ep_right.at<double>(2), 0, -ep_right.at<double>(0), 
						-ep_right.at<double>(1), ep_right.at<double>(0), 0 );


			ep_right2.at<double>(0,1) = -ep_right.at<double>(2,0);
			ep_right2.at<double>(0,2) = ep_right.at<double>(1,0);
			ep_right2.at<double>(1,0) = ep_right.at<double>(2);
			ep_right2.at<double>(1,2) = -ep_right.at<double>(0);
			ep_right2.at<double>(2,0) = -ep_right.at<double>(1);
			ep_right2.at<double>(2,1) = ep_right.at<double>(0);

			ep_right2O.at<double>(0,1) = -ep_rightO.at<double>(2,0); //SZUKANIE BLEDU
			ep_right2O.at<double>(0,2) = ep_rightO.at<double>(1,0); //SZUKANIE BLEDU
			ep_right2O.at<double>(1,0) = ep_rightO.at<double>(2); //SZUKANIE BLEDU
			ep_right2O.at<double>(1,2) = -ep_rightO.at<double>(0); //SZUKANIE BLEDU
			ep_right2O.at<double>(2,0) = -ep_rightO.at<double>(1); //SZUKANIE BLEDU
			ep_right2O.at<double>(2,1) = ep_rightO.at<double>(0); //SZUKANIE BLEDU



			cv::Mat P_lp (3,3, CV_64FC1);
			cv::Mat P_lp2 (3,4, CV_64FC1);
			cv::Mat P_rp (3,3, CV_64FC1);
			cv::Mat P_rp2 (3,4, CV_64FC1);
			
			cv::Mat P_lpO (3,3, CV_64FC1); //SZUKANIE BLEDU
			cv::Mat P_lp2O (3,4, CV_64FC1); //SZUKANIE BLEDU
			cv::Mat P_rpO (3,3, CV_64FC1); //SZUKANIE BLEDU
			cv::Mat P_rp2O (3,4, CV_64FC1); //SZUKANIE BLEDU

			P_lp = (ep_left2 * fundemental);
			P_rp = (ep_right2 * fundemental);

			P_lpO = (ep_left2O * fundemental);//SZUKANIE BLEDU
			P_rpO = (ep_right2O * fundemental);//SZUKANIE BLEDU

			//std::cout << "Macierz fundemental: \n" << fundemental << std::endl;
			//std::cout << "Macierz ep_left: \n" << ep_left << std::endl;
			//std::cout << "Macierz ep_left2: \n" << ep_left2 << std::endl;
			//std::cout << "Macierz ep_right2: \n" << ep_right2 << std::endl;

			for (int i = 0; i < 3; i++) {
				P_lp.col(i).copyTo(P_lp2.col(i));
				P_rp.col(i).copyTo(P_rp2.col(i));

				P_lpO.col(i).copyTo(P_lp2O.col(i));//SZUKANIE BLEDU
				P_rpO.col(i).copyTo(P_rp2O.col(i));//SZUKANIE BLEDU
			}

			ep_left.copyTo(P_lp2.col(3));		
			ep_right.copyTo(P_rp2.col(3));	

			ep_leftO.copyTo(P_lp2O.col(3));//SZUKANIE BLEDU		
			ep_rightO.copyTo(P_rp2O.col(3));//SZUKANIE BLEDU	

			float m[3][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}};
			cv::Mat Pr(3,4, CV_32FC1, m);
			//Pr = (1, 0, 0, 0, 
			//	  0, 1, 0, 0,
			//	  0, 0, 1, 0);

			cv:: Mat poTriangulacji(1, keypointsl.size(), CV_32FC4);
			cv:: Mat dst(1, selPointsLeft.size(), CV_32FC1);
			std::vector<cv::Point3f> dsst;
			
			std::cout << "Pr: \n" << Pr << std::endl;
			std::cout << "P_rp2: \n" << P_rp2 << std::endl;
			std::cout << "P_lp2: \n" << P_lp2 << std::endl;
			//std::cout << "P_rp2O: \n" << P_rp2O << std::endl; //SZUKANIE BLEDU
			std::cout << "selPointsLeft: \n" << selPointsLeft << std::endl;
			std::cout << "selPointsRight: \n" << selPointsRight << std::endl;
			cv::triangulatePoints(Pr, P_rp2, selPointsLeft, selPointsRight, poTriangulacji);
			std::cout << "poTriangulacji: \n" << poTriangulacji << std::endl;
			//std::cout << "Macierz Pr: \n" << Pr << std::endl;
			//std::cout << "Macierz P_rp2: \n" << P_rp2 << std::endl;
			//std::cout << "Macierz selPointsLeft: \n" << selPointsLeft << std::endl;
			//std::cout << "Macierz selPointsRight: \n" << selPointsRight << std::endl;
			//std::cout << "Macierz fundemental: \n" << fundemental << std::endl;
			
			
			//std::cout << "Macierz potriangulacji- col 0: \n" << poTriangulacji.col(0)<< std::endl;
		//	std::cout << "Macierz potriangulacji- col 1 \n" << poTriangulacji.col(1)<< std::endl;
			
			cv::Mat Konwersja[] = {poTriangulacji, poTriangulacji, poTriangulacji,poTriangulacji};
			cv::Mat poTriangulacjiKonw(1, keypointsl.size(), CV_32FC4);
			int n = poTriangulacjiKonw.checkVector(4);

			//std::cout << "Macierz potriangulacji i konw- col 0: \n" << poTriangulacjiKonw.col(0)<< std::endl;
		//	cv::Mat wynikowa = konwersjaZHomo(poTriangulacji);
		//	std:: cout << "wynikowa = " << wynikowa << std::endl;
	}

	


	//perspectiveTransform() do pokazania 3d
	return 0;
}