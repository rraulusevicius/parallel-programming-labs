package main

import (
	"fmt"
	"math/rand"
	"sync"
)

func randRange(min, max int) int {
	return rand.Intn(max-min) + min
}

func sensor(fromSenders chan int, stopSignal chan struct{}, wg *sync.WaitGroup, sensorMin int, sensorMax int, sensorID int, count int) {
	defer wg.Done()

	for i := 0; i < count; i++ {
		temperature := randRange(sensorMin, sensorMax)
		select {
		case <-stopSignal:
			fmt.Printf("Sensor %d received stop signal", sensorID)
			return
		case fromSenders <- temperature:
			//fmt.Printf("Sensor %d: %d°C\n", sensorID, temperature)
		}
	}
}

func analyzer(fromSenders chan int, stopSignal chan struct{}, toCold chan int, toNormal chan int, toHot chan int, wg *sync.WaitGroup) {
	defer wg.Done()
	defer close(toCold)
	defer close(toNormal)
	defer close(toHot)

	count := 0
	cold := 0
	normal := 0
	hot := 0

	for count < 22 {
		temperature := <-fromSenders
		if temperature < 18 {
			toCold <- temperature
			//fmt.Printf("Analyzer: %d°C -> Cold\n", temperature)
			cold++
		} else if temperature >= 18 && temperature <= 24 {
			toNormal <- temperature
			//fmt.Printf("Analyzer: %d°C -> Normal\n", temperature)
			normal++
		} else {
			toHot <- temperature
			//fmt.Printf("Analyzer: %d°C -> Hot\n", temperature)
			hot++
		}
		count++
	}
	fmt.Printf("Analyzer processed %d readings, sending stop signal \n", count)
	close(stopSignal)
	fmt.Printf("Analyzer summary: Cold=%d, Normal=%d, Hot=%d\n", cold, normal, hot)
}

func coldCollector(toCold chan int, wg *sync.WaitGroup) {
	defer wg.Done()
	var temperatures []int

	for temp := range toCold {
		temperatures = append(temperatures, temp)
	}
	var average float64 = 0
	for i := range temperatures {
		average += float64(temperatures[i])
	}
	average /= float64(len(temperatures))
	fmt.Println("Cold Collector collected temperatures: ", temperatures)
	fmt.Println("Cold collector average temperature: ", average)
}

func normalCollector(toNormal chan int, wg *sync.WaitGroup) {
	defer wg.Done()
	var temperatures []int

	for temp := range toNormal {
		temperatures = append(temperatures, temp)
	}
	var average float64 = 0
	for i := range temperatures {
		average += float64(temperatures[i])
	}
	average /= float64(len(temperatures))
	fmt.Println("Normal collector collected temperatures:", temperatures)
	fmt.Println("Normal collector average temperature: ", average)
}

func hotCollector(toHot chan int, wg *sync.WaitGroup) {
	defer wg.Done()
	var temperatures []int

	for temp := range toHot {
		temperatures = append(temperatures, temp)
	}
	var average float64 = 0
	for i := range temperatures {
		average += float64(temperatures[i])
	}
	average /= float64(len(temperatures))
	fmt.Println("Hot collector collected temperatures:", temperatures)
	fmt.Println("Hot collector average temperature: ", average)
}

func main() {
	// Channels
	fromSenders := make(chan int)
	stopSignal := make(chan struct{})
	toCold := make(chan int)
	toNormal := make(chan int)
	toHot := make(chan int)

	var wg sync.WaitGroup

	// Start collectors
	wg.Add(3)
	go coldCollector(toCold, &wg)
	go normalCollector(toNormal, &wg)
	go hotCollector(toHot, &wg)

	// Start analyzer
	wg.Add(1)
	go analyzer(fromSenders, stopSignal, toCold, toNormal, toHot, &wg)

	var sensorWg sync.WaitGroup

	// Start sensors
	sensorWg.Add(4)
	go sensor(fromSenders, stopSignal, &sensorWg, 15, 30, 1, 6)
	go sensor(fromSenders, stopSignal, &sensorWg, 15, 30, 2, 5)
	go sensor(fromSenders, stopSignal, &sensorWg, 15, 30, 3, 7)
	go sensor(fromSenders, stopSignal, &sensorWg, 15, 30, 4, 4)

	// Wait for sensors to finish
	sensorWg.Wait()
	close(fromSenders) // Close sensor/sender channel

	// Wait for rest of goroutines to finish
	wg.Wait()

	fmt.Printf("All done \n")
}
