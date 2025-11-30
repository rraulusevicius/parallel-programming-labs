package main

import (
	"fmt"
	"math/rand"
	"sync"
)

func sender1(ch chan int, stopSignal chan struct{}, wg *sync.WaitGroup) {
	defer wg.Done()

	for i := 0; i < 8; i++ {
		number := rand.Intn(100)
		select {
		case <-stopSignal:
			fmt.Printf("Sender 1 received stop signal. Exiting. \n")
			return
		case ch <- number:
			fmt.Printf("Sender  sent: %d\n", number)
		}
	}
}

func sender2(ch chan int, stopSignal chan struct{}, wg *sync.WaitGroup) {
	defer wg.Done()

	for i := 0; i < 7; i++ {
		number := rand.Intn(100)
		select {
		case <-stopSignal:
			fmt.Printf("Sender 2 received stop signal. Exiting. \n")
			return
		case ch <- number:
			fmt.Printf("Sender 2 sent: %d\n", number)
		}
	}
}

func sender3(ch chan int, stopSignal chan struct{}, wg *sync.WaitGroup) {
	defer wg.Done()

	for i := 0; i < 10; i++ {
		number := rand.Intn(100)
		select {
		case <-stopSignal:
			fmt.Printf("Sender 3 received stop signal. Exiting. \n")
			return
		case ch <- number:
			fmt.Printf("Sender 3 sent: %d\n", number)
		}
	}
}

func receiver(fromSenders chan int, stopSignal chan struct{}, toDivide3 chan int, toDivide5 chan int, wg *sync.WaitGroup) {
	defer wg.Done()
	defer close(toDivide3)
	defer close(toDivide5)
	count := 0

	for count < 25 {
		number := <-fromSenders
		fmt.Printf("Receiver got: %d\n", number)
		if number%3 == 0 {
			toDivide3 <- number
			fmt.Printf("Receiver sent %d to toDivide3 channel\n", number)
		} else if number%5 == 0 {
			toDivide5 <- number
			fmt.Printf("Receiver sent %d to toDivide5 channel\n", number)
		}
		count++
	}
	fmt.Printf("Receiver received %d numbers. Sending stop signal to senders. \n", count)
	close(stopSignal)
}

func printer3(toDivide3 chan int, wg *sync.WaitGroup) {
	defer wg.Done()
	var numbers []int

	for number := range toDivide3 {
		numbers = append(numbers, number)
	}

	fmt.Println("Numbers received that are divisible by 3:", numbers)
}

func printer5(toDivide5 chan int, wg *sync.WaitGroup) {
	defer wg.Done()
	var numbers []int

	for number := range toDivide5 {
		numbers = append(numbers, number)
	}

	fmt.Println("Numbers received that are divisible by 5:", numbers)
}

func main() {
	//Channels
	fromSenders := make(chan int)
	toDivide3 := make(chan int)
	toDivide5 := make(chan int)
	stopSignal := make(chan struct{})

	var wg sync.WaitGroup

	//Start printers/collectors
	wg.Add(2)
	go printer3(toDivide3, &wg)
	go printer5(toDivide5, &wg)

	//Start Receiver
	wg.Add(1)
	go receiver(fromSenders, stopSignal, toDivide3, toDivide5, &wg)

	//Make sender waitgroup
	var sendersWg sync.WaitGroup

	//Start senders
	sendersWg.Add(3)
	go sender1(fromSenders, stopSignal, &sendersWg)
	go sender2(fromSenders, stopSignal, &sendersWg)
	go sender3(fromSenders, stopSignal, &sendersWg)

	//Wait for senders to finish
	sendersWg.Wait()
	close(fromSenders) // Send closing singal

	//Wait for all goroutines to finish
	wg.Wait()

	fmt.Printf("All done!\n")
}
