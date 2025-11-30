package main

import (
	"fmt"
	"sync"
)

func sender1(ch chan int, stopSignal chan struct{}, wg *sync.WaitGroup) {
	defer wg.Done()
	for i := 0; i < 100; i++ {
		select {
		case <-stopSignal:
			fmt.Printf("Sender 1 received stop signal, exiting \n")
			return
		case ch <- i:
			fmt.Printf("Sender 1 sent: %d\n", i)
		}
	}
}

func sender2(ch chan int, stopSignal chan struct{}, wg *sync.WaitGroup) {
	defer wg.Done()
	for i := 11; i < 100; i++ {
		select {
		case <-stopSignal:
			fmt.Printf("Sender 2 received stop signal, exiting \n")
			return
		case ch <- i:
			fmt.Printf("Sender 2 sent: %d\n", i)
		}
	}
}

func receiver(fromSenders chan int, toEven chan int, toOdd chan int, stopSignal chan struct{}, wg *sync.WaitGroup) {
	defer wg.Done()
	defer close(toEven)
	defer close(toOdd)

	count := 0

	for count < 20 {
		num := <-fromSenders
		if num%2 == 0 {
			toEven <- num
			fmt.Printf("Receiver sent to even: %d\n", num)
		} else {
			toOdd <- num
			fmt.Printf("Receiver sent to odd: %d\n", num)
		}
		count++
	}
	fmt.Printf("Receiver processed 20 numbers, sending stop signal \n")
	close(stopSignal) // Signal senders to stop

}

func printerEven(ch chan int, wg *sync.WaitGroup) {
	defer wg.Done()
	var numbers []int

	for num := range ch {
		numbers = append(numbers, num)
	}

	fmt.Println("Even numbers:", numbers)
}

func printerOdd(ch chan int, wg *sync.WaitGroup) {
	defer wg.Done()
	var numbers []int

	for num := range ch {
		numbers = append(numbers, num)
	}

	fmt.Println("Odd numbers:", numbers)
}

func main() {
	// Channels
	fromSenders := make(chan int)
	toEven := make(chan int)
	toOdd := make(chan int)
	stopSignal := make(chan struct{})

	var wg sync.WaitGroup

	wg.Add(1)
	go printerEven(toEven, &wg)

	wg.Add(1)
	go printerOdd(toOdd, &wg)

	wg.Add(1)
	go receiver(fromSenders, toEven, toOdd, stopSignal, &wg)

	var sendersWg sync.WaitGroup

	sendersWg.Add(2)
	go sender1(fromSenders, stopSignal, &sendersWg)
	go sender2(fromSenders, stopSignal, &sendersWg)

	sendersWg.Wait()
	close(fromSenders)

	wg.Wait()

	fmt.Printf("All done!\n")
}
