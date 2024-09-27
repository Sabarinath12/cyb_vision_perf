#include <opencv2/opencv.hpp>
#include <iostream>
#include <unistd.h>
#include <sys/sysinfo.h>
#include <fstream>
#include <sstream>
#include <string>
#include <array>
#include <thread>
#include <mutex>
#include <chrono>
#include <iomanip>

using namespace std;
using namespace cv;

static long double prevTotal = 0, prevIdle = 0;
static float cpuUsage = 0.0f;
static string netStatus = "Disconnected"; // Shared network status
static string batteryStatus = "Unknown"; // Battery status
static mutex netMutex; // Mutex for thread safety

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

// Function to check network connection status by pinging google.com
void pingNetwork() {
    while (true) {
        const char* cmd = "ping -c 1 google.com > /dev/null 2>&1"; // Ping once and suppress output
        int result = system(cmd); // Execute the command

        string currentStatus = (result == 0) ? "Connected" : "Disconnected";

        {
            lock_guard<mutex> lock(netMutex); // Lock mutex for thread safety
            netStatus = currentStatus; // Update the shared network status
        }

        std::this_thread::sleep_for(std::chrono::seconds(5)); // Ping every 5 seconds
    }
}


// Function to get current date and time
string getCurrentDateTime() {
    auto now = chrono::system_clock::now();
    time_t now_c = chrono::system_clock::to_time_t(now);
    tm now_tm = *localtime(&now_c);

    stringstream ss;
    ss << put_time(&now_tm, "%Y-%m-%d %H:%M:%S"); // Format: YYYY-MM-DD HH:MM:SS
    return ss.str();
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

    // Start the network ping thread
    thread pingThread(pingNetwork);

    Mat frame;
    int frameSkip = 3; // Skip every 3 frames for processing
    int frameCount = 0;

    while (true) {
        capture >> frame;
        if (frame.empty()) break;

        // Resize the frame for faster processing
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

        // Update CPU usage and get battery status
        if (frameCount % (frameSkip * 5) == 0) {
            cpuUsage = getCPUUsage(); // Update every few frames
            getBatteryStatus(); // Get battery status periodically
        }

        // Get RAM usage
        float ramUsage = getRAMUsage();

        // Get the latest network status
        {
            lock_guard<mutex> lock(netMutex); // Lock mutex to read shared status
            // Prepare display text
            stringstream cpuText, ramText;
            cpuText << "CPU: " << fixed << setprecision(2) << cpuUsage << "%";
            ramText << "RAM: " << fixed << setprecision(2) << ramUsage << "%";
            string netText = "Network: " + netStatus;
            string dateTimeText = getCurrentDateTime();

            // Display text on frame
            putText(resizedFrame, cpuText.str(), Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);
            putText(resizedFrame, ramText.str(), Point(10, 40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);
            putText(resizedFrame, netText, Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);
            putText(resizedFrame, dateTimeText, Point(resizedFrame.cols - 200, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);
        }

        // Display the resulting frame
        imshow("Red-Tinted Face Detection", resizedFrame);

        // Break the loop on 'q' key press
        if (waitKey(10) == 'q') break;

        frameCount++;
        usleep(50000); // Introduce a small delay (50ms)
    }

    // Clean up
    pingThread.detach(); // Detach the ping thread (if you want to let it run until the program ends)
    capture.release();
    destroyAllWindows();
    return 0;
}
