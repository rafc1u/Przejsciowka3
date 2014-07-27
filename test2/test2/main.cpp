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

template <class type>
type mediana(std::vector<type> &v);

enum nachylenie{
			
				brak = 0,
				dodatnie = 1,
				ujemne = 2,
			};

void Ocena(std::vector<cv::DMatch> matches, std::vector<cv::KeyPoint> keypointsl, std::vector<cv::KeyPoint> keypointsr){

	std::vector<cv::DMatch>::iterator itDM = matches.begin();
	std::vector<double> odleglosci;
	std::vector<nachylenie> vnachyl;
	
	std::cout << "\nParametry kolejnych par: " << std::endl;
	for (int i = 0; i < matches.size(); i++){
		double dlug;
		cv::Point left = keypointsl.at(itDM->queryIdx).pt;
		cv::Point right = keypointsr.at(itDM->trainIdx).pt;
		dlug = sqrt(double(double(left.x-right.x)*double(left.x-right.x)) + double(double(left.y-right.y)*double(left.y-right.y))  );
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

void Filtrowanie (std::vector<int> pointIndexesLeft, std::vector<int> pointIndexesRight, 
							std::vector<cv::KeyPoint> keypointsl, std::vector<cv::KeyPoint> keypointsr,
							std::vector<cv::DMatch> matches, cv::Mat imagel, cv::Mat imager){

				std::vector<int>::const_iterator itl = pointIndexesLeft.begin();
				std::vector<int>::const_iterator itr = pointIndexesRight.begin();
				std::vector<double> odleglosci;
				std::vector<nachylenie> vnachyl;
				cv::Mat imageMatchesPrzed, imageMatchesPo;
				std::vector<cv::DMatch> matchesPrawidlowe;

				Ocena(matches,keypointsl, keypointsr);

				//Wyliczenie mediany odleglosci laczacych punkty charakterystyczne oraz wspó³czynniki nachylenia tych prostych
				double mOdleglosc = mediana(odleglosci);
				std::cout << "Mediana odleglosci punktow char. = " << mOdleglosc << std::endl;
				nachylenie mNachylenie = mediana(vnachyl);
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
				cv::drawMatches(imagel,keypointsl,imager, keypointsr, matches, imageMatchesPrzed);
				cv::imshow("Przed filtracja", imageMatchesPrzed);

				//Proces odfiltrowania
				itl = pointIndexesLeft.begin();
				itr = pointIndexesRight.begin();
				

				std::vector<double>::iterator itOdl = odleglosci.begin();
				std::vector<nachylenie>::iterator itNach = vnachyl.begin();

				for ( int i = 0; i < odleglosci.size(); i ++ ){

					if (*itOdl == mOdleglosc && *itNach == mNachylenie){
						
						cv::DMatch dmatch;
						dmatch.queryIdx = *itl;
						dmatch.trainIdx = *itr;
						matchesPrawidlowe.push_back(dmatch);
						
					}

					itOdl++;
					itNach++;
					itl++;
					itr++;
				}
				// rysowanie DOPASOWAN Po FILTRACJI
				cv::drawMatches(imagel,keypointsl,imager, keypointsr, matchesPrawidlowe, imageMatchesPo);
				cv::imshow("Pofiltracja", imageMatchesPo);
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


	cv::Mat konwersjaZHomo(cv::Mat wej){
		//int typ = wej.type();
		const int rows = wej.rows;
		const int cols = wej.cols;

		//float tmp[wejRows][wejCols];

		//allocate the array
		//float** arr = new float*[rows -1];
		//	for(int i = 0; i < rows-1; i++)
		//	arr[i] = new float[cols];


		float arr[3][7];
		for (int i = 0; i < cols; i++ ){

			for (int j =0; j < rows -1; j++ ){

				float a = wej.at<float>(j,i) / wej.at<float>(rows-1, i);
			//	std::cout << "kolumna: " << i << " linia: " << j << "Dzielenie: " << wej.at<float>(j,i) << " / " << wej.at<float>(rows-1, i) << std::endl;
				arr[j][i] = a;
			//	std::cout<< " = " << arr[j][i] << std::endl;
			}
		}

		cv::Mat wynik(rows -1,cols, CV_32FC1, arr);
		std::cout << "wynikowa z fkc " << wynik << std::endl;

		//deallocate the array
		//for(int i = 0; i < rows -1 ; i++)
		//	 delete[] arr[i];
		//	 delete[] arr;

		return wynik;
	}


int main(){


	cv:: Mat zdjl, zdjr, imagel, imager, outImgl, outImgr, matchess, matchess2;
	int wartSlider = 18000; //28000 uzywac
	std::vector<cv::KeyPoint> keypointsl, keypointsr;;

	cv::SurfDescriptorExtractor surfDesc;
	cv::Mat descriptorsl, descriptorsr;

	//zdjl = cv::imread("C:\\left.jpg", CV_LOAD_IMAGE_COLOR );
	//zdjr = cv::imread("C:\\right.jpg", CV_LOAD_IMAGE_COLOR );

	//zdjl = cv::imread("C:\\Users\\kkk\\Documents\\Visual Studio 2010\\Projects\\HelloGUIWorld2\\Win32\\Debug\\lewa00.jpg", CV_LOAD_IMAGE_COLOR );
	//zdjr = cv::imread("C:\\Users\\kkk\\Documents\\Visual Studio 2010\\Projects\\HelloGUIWorld2\\Win32\\Debug\\prawa00.jpg", CV_LOAD_IMAGE_COLOR );

	zdjl = cv::imread("C:\\Users\\kkk\\Desktop\\leftp2.jpg", CV_LOAD_IMAGE_COLOR );
	zdjr = cv::imread("C:\\Users\\kkk\\Desktop\\rightp2.jpg", CV_LOAD_IMAGE_COLOR );

	zdjl = cv::imread("C:\\Users\\kkk\\Desktop\\dol.jpg", CV_LOAD_IMAGE_COLOR );
	zdjr = cv::imread("C:\\Users\\kkk\\Desktop\\gora.jpg", CV_LOAD_IMAGE_COLOR );

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

			enum nachylenie{
			
				brak = 0,
				dodatnie = 1,
				ujemne = 2,
			};

			std::vector<nachylenie> vnachyl;	
			//BLAD- przeciez keypointy lewe i prawe powinny byc brane z DMatch! a nie po koleji!
			//Napisaac Funkcje filtrowanie(pointIndexesLeft, pointIndexesRight, keyointsl, keypointsr)

			

		/*
		//filtrowanie
		for (int i = 0; i < pointIndexesLeft.size(); i++){

			double wynik;
			cv::Point left;
			cv::Point right;
			int abc = *itl;
			left = keypointsl.at(*itl).pt;
			right = keypointsr.at(*itr).pt;
			// to liczenie powoduje blad przy dziel przez 0
			wynik = tan( (double) (double(left.y-right.y)/(double(left.x-right.x)) ) );
			//uzywac tego
			wynik = sqrt(double(double(left.x-right.x)*double(left.x-right.x))+double(double(left.y-right.y)*double(left.y-right.y))  );
			odl_eu.push_back( wynik );
			if (left.y < right.y){
				vnachyl.push_back(dodatnie);
			}
			else if (left.y == right.y){
				vnachyl.push_back(brak);	
			}
			else{
				vnachyl.push_back(ujemne);
			}


			
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
		int xyxx;
		nachylenie medNachylenia = mediana(vnachyl);

		int mediana = median(odl_eu);
		pp = 0;
		for (int i = 0; i < odl_eu.size(); i++){
			
			//wyrzucenie POWINNA BYC MEDIANA
			if (odl_eu.at(i) != mediana || vnachyl.at(i) != medNachylenia){
				int wyp = i-pp;
				matches2.erase(matches2.begin()+wyp);
				//pointIndexesLeft2.erase(pointIndexesLeft2.begin() + wyp);
				//pointIndexesRight2.erase(pointIndexesRight2.begin() + wyp);
				//czy napewno?
				//keypointsl2.erase(keypointsl2.begin() + i);
				//keypointsr2.erase(keypointsr2.begin() + i);
				pp++;
			}

		}

		*/
		Filtrowanie(pointIndexesLeft, pointIndexesRight, keypointsl, keypointsr, matches, imagel, imager);

		


		//pointIndexes - przypisanie poprawnych
			pointIndexesLeft = pointIndexesLeft2;
			pointIndexesRight = pointIndexesRight2;

		// Convert keypoints into Point2f
			std::vector<cv::Point2f> selPointsLeft, selPointsRight, vectest;
			
			cv::KeyPoint::convert(keypointsl,selPointsLeft,pointIndexesLeft);
			cv::KeyPoint::convert(keypointsr,selPointsRight,pointIndexesRight);

			
			// rysowanie DOPASOWAN
			cv::drawMatches(imagel,keypointsl,imager, keypointsr, matches2, matchess);
			//imshow("Po filtracji", matchess);

			// rysowanie DOPASOWAN
			//cv::drawMatches(imagel,keypointsl,imager, keypointsr, matches2, matchess);
			//imshow("Matches with correct function", matchess2);

			//Wyliczenie macierzy fundemental
			cv::Mat fundemental= cv::findFundamentalMat(
            cv::Mat(selPointsLeft), // points in first image
            cv::Mat(selPointsRight), // points in second image
            CV_FM_RANSAC);       // 8-point method
			std::cout << "fundemental" << fundemental << std::endl;
			cv::Mat l_prim;
			//l_prim = fundemental*
			
			//sprawdzenie wg wzoru 11.2 ze strony 279 z MVG
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
			std::cout << "t " << t << std::endl;


			cv::Mat(H1);
			cv::Mat(H2);

			cv::stereoRectifyUncalibrated(selPointsLeft, selPointsRight, fundemental, imagel.size(), H1,H2);

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