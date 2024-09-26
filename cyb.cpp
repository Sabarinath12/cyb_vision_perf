#include <opencv2/opencv.hpp>
#include <iostream>
#include <unistd.h>
#include <sys/sysinfo.h>
#include <fstream>
#include <sstream>
#include <string>

using namespace std;
using namespace cv;

static long double prevTotal = 0, prevIdle = 0;
static float cpuUsage = 0.0f;

// Function to get CPU usage
float getCPUUsage() {
    long double totalIdle = 0, totalCpu = 0;

    ifstream statFile("/proc/stat");
    if (!statFile.is_open()) {
        cerr << "Could not open /proc/stat" << endl;
        return cpuUsage; // Return last known value if failed
    }

    string line;
    getline(statFile, line);
    statFile.close();

    istringstream ss(line);
    string cpu;
    long double user, nice, system, idle, iowait, irq, softirq, steal;

    ss >> cpu >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal;

    totalCpu = user + nice + system + idle + iowait + irq + softirq + steal;
    totalIdle = idle;

    cpuUsage = 100.0 * (1 - (totalIdle - prevIdle) / (totalCpu - prevTotal));
    prevTotal = totalCpu;
    prevIdle = totalIdle;

    return cpuUsage;
}

// Function to get accurate RAM usage
float getRAMUsage() {
    ifstream meminfo("/proc/meminfo");
    string line;
    long long total = 0, free = 0, available = 0;

    while (getline(meminfo, line)) {
        if (line.find("MemTotal:") == 0) {
            sscanf(line.c_str(), "MemTotal: %lld kB", &total);
        } else if (line.find("MemFree:") == 0) {
            sscanf(line.c_str(), "MemFree: %lld kB", &free);
        } else if (line.find("MemAvailable:") == 0) {
            sscanf(line.c_str(), "MemAvailable: %lld kB", &available);
        }
    }

    meminfo.close();

    if (available > 0) {
        return ((total - available) / static_cast<float>(total)) * 100.0;
    }

    return ((total - free) / static_cast<float>(total)) * 100.0;
}

// Function to check network connection status
string getNetworkStatus() {
    ifstream netFile("/proc/net/dev");
    string line;
    bool isConnected = false;

    while (getline(netFile, line)) {
        // Check if any interface has received bytes (not just loopback)
        if (line.find("lo:") == string::npos) {
            size_t pos = line.find(":");
            if (pos != string::npos) {
                string interfaceName = line.substr(0, pos);
                // Ignore empty or disconnected interfaces
                if (line.find_first_of("0123456789") != string::npos) {
                    isConnected = true;
                    break;
                }
            }
        }
    }

    netFile.close();
    return isConnected ? "Connected" : "Disconnected";
}

// Function to apply red tint using forEach
void applyRedTint(Mat& frame) {
    frame.forEach<Vec3b>([](Vec3b& pixel, const int* position) -> void {
        pixel[0] = static_cast<uchar>(pixel[0] * 0.5); // Reduce blue
        pixel[1] = static_cast<uchar>(pixel[1] * 0.5); // Reduce green
        pixel[2] = static_cast<uchar>(std::min(pixel[2] * 1.5, 255.0)); // Increase red, clamp to 255
    });
}

int main() {
    // Load the Haar Cascade for face detection
    CascadeClassifier faceCascade;
    if (!faceCascade.load("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")) {
        cerr << "Error loading Haar cascade file!" << endl;
        return -1;
    }

    // Open the default camera
    VideoCapture capture(0);
    if (!capture.isOpened()) {
        cerr << "Error opening video stream!" << endl;
        return -1;
    }

    Mat frame;
    int frameSkip = 3; // Skip every 3 frames for processing
    int frameCount = 0;

    while (true) {
        capture >> frame;
        if (frame.empty()) break;

        // Resize the frame for faster processing (adjust this as needed)
        Mat resizedFrame;
        resize(frame, resizedFrame, Size(), 1.0, 1.0); // Keep full resolution for better detection

        // Convert to grayscale for detection
        Mat gray;
        cvtColor(resizedFrame, gray, COLOR_BGR2GRAY);

        // Detect faces every nth frame
        if (frameCount % frameSkip == 0) {
            vector<Rect> faces;
            faceCascade.detectMultiScale(gray, faces, 1.1, 4, 0, Size(30, 30));

            // Draw rectangles around detected faces
            for (const auto& face : faces) {
                rectangle(resizedFrame, face, Scalar(255, 255, 255), 2); // White rectangle
            }
        }

        // Apply red tint to the frame
        applyRedTint(resizedFrame);

        // Display CPU, RAM usage, and network status
        float ramUsage = getRAMUsage();
        stringstream cpuText, ramText, netText;

        // Update CPU usage only every 500ms
        if (frameCount % (frameSkip * 5) == 0) {
            cpuUsage = getCPUUsage(); // Update every few frames
        }

        cpuText << "CPU: " << fixed << setprecision(2) << cpuUsage << "%";
        ramText << "RAM: " << fixed << setprecision(2) << ramUsage << "%";
        netText << "Network: " << getNetworkStatus();
        putText(resizedFrame, cpuText.str(), Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);
        putText(resizedFrame, ramText.str(), Point(10, 40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);
        putText(resizedFrame, netText.str(), Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);

        // Display the resulting frame
        imshow("Red-Tinted Face Detection", resizedFrame);

        // Break the loop on 'q' key press
        if (waitKey(10) == 'q') break;

        frameCount++;
        usleep(50000); // Introduce a small delay (50ms)
    }

    // Release the video capture object
    capture.release();
    destroyAllWindows();
    return 0;
}
