package main

import (
	"fmt"
	"sync"
)

func sender1(ch chan int, wg *sync.WaitGroup) {
	defer wg.Done()
	//Siuncia skaicius nuo 0 iki 10
	for i := 0; i < 10; i++ {
		ch <- i
		fmt.Printf("Sender 1 sent: %d\n", i)
	}
}

func sender2(ch chan int, wg *sync.WaitGroup) {
	defer wg.Done()
	//Siuncia  skaicius nuo 11 iki 20
	for i := 10; i < 20; i++ {
		ch <- i
		fmt.Printf("Sender 2 sent: %d\n", i)
	}
}

func receiver(fromSenders chan int, toEven chan int, toOdd chan int, wg *sync.WaitGroup) {
	defer wg.Done()
	defer close(toEven)
	defer close(toOdd)

	count := 0
	//Priima 20 skaiciu

	for count < 20 {
		num := <-fromSenders
		fmt.Printf("Receiver received: %d\n", num)

		//Tikrina ar skaicius lyginis ar nelyginis
		if num%2 == 0 {
			toEven <- num
			fmt.Printf("Receiver sent to even channel: %d\n", num)
		} else {
			toOdd <- num
			fmt.Printf("Receiver sent to odd channel: %d\n", num)
		}

		//Padidininame count
		count++
	}

}

func printerEven(ch chan int, wg *sync.WaitGroup) {
	defer wg.Done()

	var numbers []int

	//Kaupia lyginius skaicius
	for num := range ch {
		numbers = append(numbers, num)
	}

	//Spausdina masyvo turini
	fmt.Println("\n=== EVEN PRINTER ===")
	fmt.Println("Even numbers received:", numbers)
}

func printerOdd(ch chan int, wg *sync.WaitGroup) {
	defer wg.Done()

	var numbers []int

	//Kaupia nelyginius skaicius
	for num := range ch {
		numbers = append(numbers, num)
	}

	//Spausdina masyvo turini
	fmt.Println("\n=== ODD PRINTER ===")
	fmt.Println("Odd numbers received:", numbers)
}

func main() {
	//Sukuriame kanalus komunikacijai
	fromSenders := make(chan int) //Buferizuotas kanalas siunteju
	toEven := make(chan int)      //Kanalas lyginiams skaiciams
	toOdd := make(chan int)       //Kanalas nelyginiams skaiciams

	var wg sync.WaitGroup

	//Paleidziame spausdintojus
	wg.Add(1)
	go printerEven(toEven, &wg)

	wg.Add(1)
	go printerOdd(toOdd, &wg)

	//Paleidziame gaveja
	wg.Add(1)
	go receiver(fromSenders, toEven, toOdd, &wg)

	//Paleidziame siuntejus
	var sendersWg sync.WaitGroup
	sendersWg.Add(2)

	go sender1(fromSenders, &sendersWg)
	go sender2(fromSenders, &sendersWg)

	//Laukiame kol siuntejai baigs
	sendersWg.Wait()
	close(fromSenders) //Uzdarome siunteju kanala

	//Laukiame visu goroutine
	wg.Wait()

	fmt.Println("\n=== PROGRAM END ===")
}
