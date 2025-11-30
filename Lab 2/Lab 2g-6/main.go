package main

import (
	"fmt"
	"sync"
)

func sender(id int, toCalculator chan int, stopSignal chan struct{}, wg *sync.WaitGroup) {
	defer wg.Done()
	
	number := id // Pirmasis skaičius = id
	
	for {
		select {
		case <-stopSignal:
			fmt.Printf("Sender %d received stop signal\n", id)
			return
		case toCalculator <- number:
			fmt.Printf("Sender %d sent: %d\n", id, number)
			// Kitas skaičius: number^2 - 4*number + 1
			number = number*number - 4*number + 1  // ✅ PATAISYTA
		}
	}
}

func calculator(fromSenders chan int, toReceiver1 chan float64, toReceiver2 chan float64, toReceiver3 chan float64, stopSignal chan struct{}, wg *sync.WaitGroup) {
	defer wg.Done()
	defer close(toReceiver1)
	defer close(toReceiver2)
	defer close(toReceiver3)
	
	window := make([]int, 0, 15)
	
	for {
		select {
		case <-stopSignal:
			fmt.Println("Calculator received stop signal")
			return
		case num, ok := <-fromSenders:
			if !ok {
				return
			}
			
			window = append(window, num)
			
			if len(window) > 15 {
				window = window[1:]
			}
			
			sum := 0
			for _, v := range window {
				sum += v
			}
			average := float64(sum) / 15.0
			
			fmt.Printf("Calculator: received %d, window size: %d, average: %.2f\n", num, len(window), average)
			
			if average > 200 {
				fmt.Printf("Calculator: average %.2f exceeded 200, sending stop signal\n", average)
				close(stopSignal)
				return
			}
			
			if average < 10 {
				toReceiver1 <- average
			}
			if average >= 0 && average <= 100 {
				toReceiver2 <- average
			}
			if average >= 75 && average <= 200 {
				toReceiver3 <- average
			}
		}
	}
}

func receiver(id int, ch chan float64, wg *sync.WaitGroup) {
	defer wg.Done()
	
	for value := range ch {
		fmt.Printf("=== RECEIVER %d received: %.2f ===\n", id, value)
	}
	
	fmt.Printf("Receiver %d finished\n", id)
}

func main() {
	toCalculator := make(chan int, 50)
	toReceiver1 := make(chan float64)
	toReceiver2 := make(chan float64)
	toReceiver3 := make(chan float64)
	stopSignal := make(chan struct{})
	
	var wg sync.WaitGroup
	
	wg.Add(3)
	go receiver(1, toReceiver1, &wg)
	go receiver(2, toReceiver2, &wg)
	go receiver(3, toReceiver3, &wg)
	
	wg.Add(1)
	go calculator(toCalculator, toReceiver1, toReceiver2, toReceiver3, stopSignal, &wg)
	
	var sendersWg sync.WaitGroup
	sendersWg.Add(10)
	
	for i := 0; i < 10; i++ {
		go sender(i, toCalculator, stopSignal, &sendersWg)
	}
	
	sendersWg.Wait()
	close(toCalculator)
	
	wg.Wait()
	
	fmt.Println("\n=== All done! ===")
}