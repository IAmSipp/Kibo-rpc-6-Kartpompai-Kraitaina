package jp.jaxa.iss.kibo.rpc.sampleapk;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import gov.nasa.arc.astrobee.Kinematics;
import gov.nasa.arc.astrobee.Result;
import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;

import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;

import org.opencv.android.Utils;
import org.opencv.aruco.Dictionary;
import org.opencv.aruco.Aruco;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Class meant to handle commands from the Ground Data System and execute them in Astrobee.
 */

public class YourService extends KiboRpcService {
    // ------- Image templates --------
    int TEMPLATE_SIZE = 8;
    String[] TEMPLATE_IMAGES_NAME = {
            "key.png",
            "compass.png",
            "coral.png",
            "fossil.png",
            "coin.png",
            "letter.png",
            "shell.png",
            "treasure_box.png"
    };

    @Override
    protected void runPlan1() {
        // Load template images
         Mat[] templateImages = LoadTemplateImages(TEMPLATE_IMAGES_NAME, TEMPLATE_SIZE);

        // ------- Area 1 --------
        Area targetArea1 = new Area(new Point(10.42, -10.58, 4.82), new Point(11.48, -10.58, 5.57));

        api.startMission();

        // Move to area 1
        Point pos = new Point(Center(10.42, 11.48), -10.00, Center(4.82, 5.57));
        Quaternion direction = Direction(-90, 0, 0, 1);
        MyMove(pos, direction, false);
        // Get item
        Mat image = TakePhoto();
        List<Mat> corners = new ArrayList<>();
        DetectAR(image, corners);
        Point itemImage1RelativeToCam = EstimatePoseFromAR(corners);

        if (itemImage1RelativeToCam != null) {
            pos = new Point(pos.getX() + itemImage1RelativeToCam.getX(), pos.getY(), pos.getZ()+itemImage1RelativeToCam.getY());
        }
        MyMove(pos, direction, false);
        image = TakePhoto();
        Mat undistortImage = UndistortImage(image);
        Mat cropImage = cropWithMargin(undistortImage, corners);
        Log.i("Crop", "Finished crop");
        api.saveMatImage(image, "Image");
        api.saveMatImage(undistortImage, "Undistort Image");
        api.saveMatImage(cropImage, "Croped Image");

        MatchTargetImage(image, templateImages, 1);

        api.reportRoundingCompletion();
        api.notifyRecognitionItem();
        api.takeTargetItemSnapshot();
    }

    private void MatchTargetImage(Mat image, Mat[] templateImages, int areaId) {
        int[] templateMatchCount = new int[TEMPLATE_SIZE];

        // Constants
        int widthMin = 10;
        int widthMax = 100;
        int changeWidth = 10;
        int changeAngle = 30;
        double threshold = 0.7;

        for (int tempNum = 0; tempNum < TEMPLATE_SIZE; tempNum++) {
            Mat template = templateImages[tempNum];
            Set<String> matchSet = new HashSet<>();

            for (int width = widthMin; width <= widthMax; width += changeWidth) {
                Mat resized = ResizingImage(template, width);

                for (int angle = 0; angle < 360; angle += changeAngle) {
                    Mat rotated = RotatingImage(resized, angle);

                    Mat result = new Mat();
                    Imgproc.matchTemplate(image, rotated, result, Imgproc.TM_CCOEFF_NORMED);

                    Core.MinMaxLocResult mmlr = Core.minMaxLoc(result);
                    if (mmlr.maxVal >= threshold) {
                        // Round match point for deduplication
                        int x = (int) Math.round(mmlr.maxLoc.x);
                        int y = (int) Math.round(mmlr.maxLoc.y);
                        matchSet.add(x + "_" + y);
                    }
                }
            }

            int matchCount = matchSet.size();
            templateMatchCount[tempNum] = matchCount;
            Log.i("Match Debug", "Template " + tempNum + " match count: " + matchCount);
        }

        int mostMatchTemplateNum = GetMaxIndex(templateMatchCount);
        String itemName = TEMPLATE_IMAGES_NAME[mostMatchTemplateNum].replace(".png", "");
        int itemCount = templateMatchCount[mostMatchTemplateNum];

        Log.i("Set Area", "SET AREA " + areaId);
        api.setAreaInfo(areaId, itemName, itemCount);
    }

    private Mat[] LoadTemplateImages(String[] TEMPLATE_IMAGES_NAME, int TEMPLATE_SIZE) {
        Mat[] templateImages = new Mat[TEMPLATE_SIZE];
        for (int i = 0; i < TEMPLATE_SIZE; i++) {
            try {
                InputStream inputStream = getAssets().open(TEMPLATE_IMAGES_NAME[i]);
                Bitmap bit = BitmapFactory.decodeStream(inputStream);
                Mat img = new Mat();
                Utils.bitmapToMat(bit, img);

                Imgproc.cvtColor(img, img, Imgproc.COLOR_BGR2GRAY);

                templateImages[i] = img;

                inputStream.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        return templateImages;
    }

    private Point EstimatePoseFromAR(List<Mat> corners) {
        Dictionary dictionary = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);

        // Prepare camera parameters
        Mat cameraMatrix = new Mat(3, 3, CvType.CV_64F);
        cameraMatrix.put(0, 0, api.getNavCamIntrinsics()[0]);  // fx, fy, cx, cy

        Mat distCoeffs = new Mat(1, 5, CvType.CV_64F);
        distCoeffs.put(0, 0, api.getNavCamIntrinsics()[1]);

        // Output rotation and translation vectors
        Mat rvec = new Mat();
        Mat tvec = new Mat();

        float markerLength = 0.05f;

        Aruco.estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs, rvec, tvec);

        if (!tvec.empty()) {
            double[] t = new double[3];
            tvec.get(0, 0, t);

            // Get position of marker in camera coordinate
            return new Point(t[0], t[1], t[2]);
        }

        return null;
    }

    private Mat TakePhoto() {
        // turn on the front flashlight
        api.flashlightControlFront(0.01f);

        try {
            Thread.sleep(200);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        // get a camera image
        Mat image = api.getMatNavCam();

        // turn off the front flashlight
        api.flashlightControlFront(0.0f);

        return image;
    }

    private Mat DetectAR(Mat image, List<Mat> cornersOut) {
        Mat markerIds = new Mat();

        Mat gray = new Mat();
        if (image.channels() == 3) {
            Log.i("Image channels", "Image is not gray scale");
            Imgproc.cvtColor(image, gray, Imgproc.COLOR_BGR2GRAY);
        } else {
            Log.i("Image channels", "Image is gray scale");
            gray = image;
        }

        // Detect AR markers
        Dictionary dictionary = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
        Aruco.detectMarkers(gray, dictionary, cornersOut, markerIds);

        return markerIds;
    }

    private Mat UndistortImage(Mat image) {
        Mat cameraMatrix = new Mat(3, 3, CvType.CV_64F);
        cameraMatrix.put(0, 0, api.getNavCamIntrinsics()[0]);

        Mat cameraCoefficients = new Mat(1, 5, CvType.CV_64F);
        cameraCoefficients.put(0, 0, api.getNavCamIntrinsics()[1]);
        cameraCoefficients.convertTo(cameraCoefficients, CvType.CV_64F);

        Mat undistortImg = new Mat();
        Calib3d.undistort(image, undistortImg, cameraMatrix, cameraCoefficients);

        return undistortImg;
    }

    private Quaternion Direction(float angle, double xd, double yd, double zd) {
        float radian = (float)((angle * Math.PI) / 180);
        float radian_cos = (float)Math.cos(radian / 2);
        float radian_sin = (float)Math.sin(radian / 2);
        float x = (float)xd;
        float y = (float)yd;
        float z = (float)zd;
        Quaternion direction = new Quaternion(x * radian_sin, y * radian_sin, z * radian_sin, radian_cos);

        Log.i("DIRECTION", "Quaternion: w=" + direction.getW()
                + ", x=" + direction.getX()
                + ", y=" + direction.getY()
                + ", z=" + direction.getZ());

        return direction;
    }

    private double Center(double a, double b) {
        return (a + b) / 2;
    }

    private static double CalculateDistance(org.opencv.core.Point p1, org.opencv.core.Point p2) {
        double dx = p1.x - p2.x;
        double dy = p1.y - p2.y;
        return Math.sqrt(Math.pow(dx, 2) + Math.pow(dy, 2));
    }

    private static List<org.opencv.core.Point> RemoveDuplicates(List<org.opencv.core.Point> points) {
        double length = 10;
        List<org.opencv.core.Point> filteredList = new ArrayList<>();

        for (org.opencv.core.Point point : points) {
            boolean include = false;
            for (org.opencv.core.Point checkPoint : filteredList) {
                double distance = CalculateDistance(point, checkPoint);

                if (distance <= length) {
                    include = true;
                    break;
                }
            }

            if (!include) {
                filteredList.add(point);
            }
        }
        return filteredList;
    }

    private int GetMaxIndex(int[] arr) {
        int maxIndex = 0;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > arr[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    private Mat cropWithMargin(Mat image, List<Mat> corners) {
        return cropWithMargin(image, corners, 150);
    }
    private Mat cropWithMargin(Mat image, List<Mat> corners, int margin) {
        List<org.opencv.core.Point> allPoints = new ArrayList<>();

        for (Mat corner : corners) {
            for (int i = 0; i < corner.rows(); i++) {
                double[] data = corner.get(i, 0);
                if (data != null && data.length == 2) {
                    allPoints.add(new org.opencv.core.Point(data[0], data[1]));
                }
            }
        }

        if (allPoints.isEmpty()) {
            Log.i("Crop", "Not detect any corners");
            return image;
        }

        Log.i("Crop", "Detect all corners");
        MatOfPoint matOfPoint = new MatOfPoint();
        Log.i("Crop", "Create new Mat of Point");
        matOfPoint.fromList(allPoints);
        Log.i("Crop", "Store all points in Mat of Point");

        Rect bbox = Imgproc.boundingRect(matOfPoint);
        Log.i("Crop", "Bounding React");

        int x = Math.max(bbox.x - margin, 0);
        int y = Math.max(bbox.y - margin, 0);
        int width = Math.min(bbox.width + 2 * margin, image.cols() - x);
        int height = Math.min(bbox.height + 2 * margin, image.rows() - y);
        Log.i("Crop", "Define marker value");
        Log.i("Crop", "x: " + x + ", y: " + y + ", width: " + width + ", height: " + height);

        try {
            Rect cropRect = new Rect(x, y, width, height);
            Log.i("Crop", "Cropping with rect: " + cropRect.toString());

            Mat cropImage = new Mat(image, cropRect);
            Log.i("Crop", "Crop image created");

            if (cropImage.channels() > 1) {
                Imgproc.cvtColor(cropImage, cropImage, Imgproc.COLOR_BGR2GRAY);
                Log.i("Crop", "Converted to gray scale");
            } else {
                Log.i("Crop", "Already grayscale, skip conversion");
            }
            Log.i("Crop", "Convert to gray scale");

            return cropImage;
        } catch (Exception e) {
            Log.e("Crop", "Exception during crop or grayscale: " + e.getMessage());
            e.printStackTrace();
            return image;
        }
    }

    private Mat ResizingImage(Mat image, int width) {
        int height = (int)(image.rows() * ((double) width/image.cols()));
        Mat resizedImage = new Mat();
        Imgproc.resize(image, resizedImage, new Size(width, height));

        return  resizedImage;
    }

    private Mat RotatingImage(Mat image, int angle) {
        org.opencv.core.Point center = new org.opencv.core.Point(image.cols() / 2.0, image.rows() / 2.0);
        Mat rotatedMat = Imgproc.getRotationMatrix2D(center, angle, 1.0);
        Mat rotatedImage = new Mat();
        Imgproc.warpAffine(image, rotatedImage, rotatedMat, image.size());

        return rotatedImage;
    }

    private void MyMove(Point point, Quaternion quaternion, boolean log_pos) {
        MyMove(point, quaternion, log_pos, 5);
    }
    private void MyMove(Point point, Quaternion quaternion, boolean log_pos, final int LOOP_MAX) {
        Result move_result = api.moveTo(point, quaternion, log_pos);

        int loopCounter = 0;
        while(!move_result.hasSucceeded() && loopCounter < LOOP_MAX){
            move_result = api.moveTo(point, quaternion, log_pos);
            ++loopCounter;
        }
    }

    class Area {
        public Point minPoint;
        public Point maxPoint;

        public Area(Point min, Point max)
        {
            minPoint = min;
            maxPoint = max;
        }
    }
}
