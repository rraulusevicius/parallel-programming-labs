#include <iostream>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <condition_variable>

using namespace std;

class Monitor {
private:
	string text;
	int vowelCount;
	condition_variable cv;
	mutex mtx;
public:
	Monitor() : text("*"), vowelCount(0) {}

	void addVowel(char c) {
		lock_guard<mutex> lock(mtx);
		text += c;
		vowelCount++;
		cv.notify_all();
	}

	void addConsonable(char c) {
		unique_lock<mutex> lock(mtx);
		cv.wait(lock, [this]() {
			return vowelCount >= 3; });
		text += c;
		vowelCount = 0;
		cv.notify_all();
	}

	string getText() {
		lock_guard<mutex> lock(mtx);
		return text;
	}
};

void workerA(Monitor& mon, atomic<int>& count, atomic<bool>& shouldStop) {
	for (int i = 0; i < 15 && !shouldStop; i++) {
		mon.addVowel('A');
		count++;
		this_thread::sleep_for(chrono::milliseconds(80));
	}
}

void workerB(Monitor& mon, atomic<int>& count, atomic<bool>& shouldStop) {
	for (int i = 0; i < 15 && !shouldStop; i++) {
		mon.addConsonable('B');
		count++;
		this_thread::sleep_for(chrono::milliseconds(100));
	}
}

void workerC(Monitor& mon, atomic<int>& count, atomic<bool>& shouldStop) {
	for (int i = 0; i < 15 && !shouldStop; i++) {
		mon.addConsonable('C');
		count++;
		this_thread::sleep_for(chrono::milliseconds(60));
	}
}

void printText(Monitor& mon, atomic<bool>& shouldStop) {

	while (!shouldStop) {
		cout << "Current line: " << mon.getText() << endl;
		this_thread::sleep_for(chrono::milliseconds(200));
	}

	cout << "Finished line: " << mon.getText() << endl;
}

int main() {
	cout << "=== Starting work. ===\n";
	
	Monitor mon;
	atomic<int> countA(0);
	atomic<int> countB(0);
	atomic<int> countC(0);
	atomic<bool> shouldStop = false;

	jthread t1(workerA, ref(mon), ref(countA), ref(shouldStop));
	jthread t2(workerB, ref(mon), ref(countB), ref(shouldStop));
	jthread t3(workerC, ref(mon), ref(countC), ref(shouldStop));
	jthread t4(printText, ref(mon), ref(shouldStop));

	while (countA < 15 && countB < 15 && countC < 15) {
		this_thread::sleep_for(chrono::milliseconds(10));
	}

	cout << "A thread: " << countA << endl;
	cout << "B thread: " << countB << endl;
	cout << "C thread: " << countC << endl;

	shouldStop = true;

	cout << "=== Finished work. ===\n";
}

class SecondMonitor {
private:
	string text;
	int vowelCount;
	mutex mtx;
	condition_variable cv;
public:
	void addVowel(char c) {
		lock_guard<mutex> lock(mtx);
		text += c;
		vowelCount++;
		cv.notify_all();
	}
	void addCons(char c) {
		unique_lock<mutex>lock(mtx);
		cv.wait(lock, [this]() { return vowelCount >= 3; });
		text += c;
		vowelCount = 0;
		cv.notify_all();
	}
	string getText() {
		lock_guard<mutex> lock(mtx);
		return text;
	}
};

void workerD(Monitor& mon, atomic<int>& count, atomic<bool>& shouldStop) {
	for (int i = 0; i < 15 && !shouldStop; i++) {
		mon.addVowel('A');
		count++;
		this_thread::sleep_for(chrono::milliseconds(80));
	}
}