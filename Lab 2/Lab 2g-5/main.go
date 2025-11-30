package main

import (
	"fmt"
	"math/rand"
	"sync"
)

func randRange(min int, max int) int {
	return rand.Intn(max-min) + 1
}

func sender(fromSender chan int, stopSignal chan struct{}, name string, count int, wg *sync.WaitGroup) {
	defer wg.Done()

	for i := 0; i < count; i++ {
		num := randRange(1, 10)
		select {
		case <-stopSignal:
			fmt.Printf("%s received stop signal, stopping. \n", name)
			return
		case fromSender <- num:
			fmt.Printf("%s sent %d to receiver.\n", name, num)
		}
	}
}

func receiver(fromSender chan int, stopSignal chan struct{}, toFail chan int, toPass chan int, toGood chan int, wg *sync.WaitGroup) {
	defer wg.Done()
	defer close(toFail)
	defer close(toPass)
	defer close(toGood)

	count := 0

	for count < 23 {
		num := <-fromSender
		if num < 4 {
			toFail <- num
			fmt.Printf("Coordinator sent %d to Fail", num)
		} else if num >= 4 && num < 7 {
			toPass <- num
			fmt.Printf("Coordinator sent %d to Pass", num)
		} else {
			toGood <- num
			fmt.Printf("Coordinator send %d to Good", num)
		}
		count++
	}
	fmt.Printf("Coordinator processed %d marks, sending stop signal.", count)
	close(stopSignal)
}

func registerFail(fromCoordinator chan int, wg *sync.WaitGroup) {
	defer wg.Done()
	var marks []int
	count := 0
	var average float64 = 0
	for num := range fromCoordinator {
		marks = append(marks, num)
		average += float64(num)
		count++
	}

	if len(marks) == 0 {
		fmt.Printf("Fail negavo pazymiu.")
		return
	}
	average /= float64(len(marks))
	fmt.Println("Fail register has these marks:", marks)
	fmt.Printf("Fail register mark average is %f and total marks are %f. \n", average, count)
}

func registerPass(fromCoordinator chan int, wg *sync.WaitGroup) {
	defer wg.Done()
	var marks []int
	count := 0
	var average float64 = 0
	for num := range fromCoordinator {
		marks = append(marks, num)
		average += float64(num)
		count++
	}

	if len(marks) == 0 {
		fmt.Printf("Pass negavo pazymiu.")
		return
	}
	average /= float64(len(marks))
	fmt.Println("Pass register has these marks:", marks)
	fmt.Printf("Pass register mark average is %f and total marks are %f. \n", average, count)
}

func registerGood(fromCoordinator chan int, wg *sync.WaitGroup) {
	defer wg.Done()
	var marks []int
	count := 0
	var average float64 = 0
	for num := range fromCoordinator {
		marks = append(marks, num)
		average += float64(num)
		count++
	}

	if len(marks) == 0 {
		fmt.Printf("Good negavo pazymiu.")
		return
	}
	average /= float64(len(marks))
	fmt.Println("Good register has these marks:", marks)
	fmt.Printf("Good register mark average is %f and total marks are %f. \n", average, count)
}
