#include <iostream>
#include <string>
#include <atomic>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <cmath>

using namespace std;

class Monitor {
private:
    int a;
    int b;
    int readCount;
    mutex mtx;
    condition_variable cv;
public:
    Monitor() : a(100), b(0), readCount(0) {}

    void modifyValues() {
        unique_lock<mutex> lock(mtx);
        cv.wait(lock, [this]() { return readCount >= 3; });
        a = a - 10;
        b = b + 10;
        readCount = 0;
        cv.notify_all();
    }

    void readValues(int& outA, int& outB) {
        unique_lock<mutex> lock(mtx);
        outA = a;
        outB = b;
        readCount++;
        //cout << "Current readCount: " << readCount << endl; // for debugging
        cv.notify_all();
    }

    // for debugging
    void getVals(int& outA, int& outB) {
        outA = a;
        outB = b;
    }
};

void printNumbers(atomic<int>& processNum, int a, int b) {
    mutex mtx;
    lock_guard<mutex> lock(mtx);
    cout << "Process number: " << processNum << ", a = " << a << ", b = " << b << endl;
}

void readerProcess(Monitor& mon, atomic<int>& processNum, atomic<bool>& shouldStop) {
    int lastA = -999;
    int lastB = -999;
    
    int currentA = 300;
    int currentB = 250;
    
    while (abs(currentA - currentB) > 20 && !shouldStop) {
        mon.readValues(currentA, currentB);
     
        if (currentA != lastA && currentB != lastB) {
            printNumbers(processNum, currentA, currentB);
            lastA = currentA;
            lastB = currentB;
        }
        this_thread::sleep_for(chrono::milliseconds(80));
    }  
}

void writerProcess(Monitor& mon, atomic<int>& processNum, atomic<bool>& shouldStop) {
    while (!shouldStop) {
        mon.modifyValues();
        //cout << "Process no. " << processNum << " modified.\n"; // for debugging
        this_thread::sleep_for(chrono::milliseconds(40));
    }
}

int main()
{
    cout << "=== Start work. ===\n";

    Monitor mon;
    atomic<int> processNum1(1);
    atomic<int> processNum2(2);
    atomic<int> processNum3(3);
    atomic<int> processNum4(4);
    atomic<int> processNum5(5);
    atomic<bool> shouldStop = false;


    jthread writerThread1(writerProcess, ref(mon), ref(processNum1), ref(shouldStop));
    jthread writerThread2(writerProcess, ref(mon), ref(processNum2), ref(shouldStop));
    jthread readerThread1(readerProcess, ref(mon), ref(processNum3), ref(shouldStop));
    jthread readerThread2(readerProcess, ref(mon), ref(processNum4), ref(shouldStop));
    jthread readerThread3(readerProcess, ref(mon), ref(processNum5), ref(shouldStop));

    // for debugging
    /*
    while (!shouldStop) {
        int a, b;
        mon.getVals(a, b);
        cout << "Main thread a and b: " << a << " " << b << endl;
        if (abs(a - b) < 20) {
            shouldStop = true;
        }
        this_thread::sleep_for(chrono::seconds(1));
    }
    */

    this_thread::sleep_for(chrono::seconds(3));
    

    cout << "=== Finished. === \n";
}
