package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"time"
)

// Product represents a single product item
type Product struct {
	Name     string  `json:"name"`
	Quantity int     `json:"quantity"`
	Price    float64 `json:"price"`
}

// ProductResult combines a product with its calculated risk score
type ProductResult struct {
	Product   Product
	RiskScore float64
}

// Message types for communication
type MessageType int

const (
	MSG_REQUEST_DATA MessageType = iota
	MSG_DATA_RESPONSE
	MSG_ADD_DATA
	MSG_ADD_RESULT
	MSG_GET_RESULTS
	MSG_RESULTS_RESPONSE
	MSG_DONE
	MSG_NO_DATA
	MSG_DATA_FULL
	MSG_DATA_ADDED
	MSG_STOP_WORKERS
)

// Message structure for inter-goroutine communication
type Message struct {
	Type      MessageType
	Product   Product
	Result    ProductResult
	Results   []ProductResult
	RequestID int
}

// calculateRiskScore performs the complex calculation from your C++ code
func calculateRiskScore(p Product) float64 {
	score := 0.0
	const ITERATIONS = 500000

	for i := 1; i <= ITERATIONS; i++ {
		baseCalc := float64(p.Quantity) * p.Price * float64(i)

		trigComponent := math.Sin(baseCalc/1000.0) * math.Cos(baseCalc/2000.0)
		logComponent := math.Log(math.Abs(baseCalc)+1) / math.Log(float64(i)+1)
		expComponent := math.Exp(-float64(i)/50000.0) * baseCalc
		sqrtComponent := math.Sqrt(math.Abs(trigComponent*logComponent)) / float64(i)
		powerComponent := math.Pow(math.Abs(baseCalc/float64(i)), 0.3)

		score += (trigComponent + logComponent + expComponent + sqrtComponent + powerComponent) / float64(i)

		if i%1000 == 0 {
			score *= 1.001
		}

		if i%100 == 0 {
			temp := score
			for j := 0; j < 10; j++ {
				temp = math.Sin(temp) + math.Cos(temp*0.1)
			}
			score += temp * 0.0001
		}
	}

	finalScore := math.Abs(score)

	nameHash := 0.0
	for _, c := range p.Name {
		nameHash += float64(c) * 3.14159
	}
	finalScore += math.Sin(nameHash) * 0.1

	return finalScore
}

// isHighRisk checks if the risk score passes the threshold
func isHighRisk(score float64) bool {
	return score > 100000000.0 // 100 million
}

// loadProductsFromFile reads products from JSON file
func loadProductsFromFile(filename string) ([]Product, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	var products []Product
	err = json.Unmarshal(data, &products)
	if err != nil {
		return nil, err
	}

	return products, nil
}

// DataThread manages the data array and responds to requests
func DataThread(dataIn chan Message, dataOut chan Message) {
	const CAPACITY = 18
	var dataArray [CAPACITY]Product
	count := 0
	front := 0
	rear := 0
	producerDone := false // ← Track if producer finished

	for {
		msg := <-dataIn
		switch msg.Type {
		case MSG_ADD_DATA:
			if count < CAPACITY {
				dataArray[rear] = msg.Product
				rear = (rear + 1) % CAPACITY
				count++
				dataOut <- Message{Type: MSG_DATA_ADDED}
			} else {
				dataOut <- Message{Type: MSG_DATA_FULL}
			}

		case MSG_REQUEST_DATA:
			if count > 0 {
				product := dataArray[front]
				front = (front + 1) % CAPACITY
				count--
				dataOut <- Message{
					Type:    MSG_DATA_RESPONSE,
					Product: product,
				}
			} else if producerDone {
				// No data AND producer finished = really done
				dataOut <- Message{Type: MSG_DONE}
			} else {
				// No data but producer still sending = wait
				dataOut <- Message{Type: MSG_NO_DATA}
			}

		case MSG_DONE:
			producerDone = true
			// Don't return yet - let workers drain remaining data
		}
	}
}

// ResultThread collects and stores results in sorted order
func ResultThread(resultIn chan Message, resultOut chan Message) {
	const MAX_RESULTS = 50 // Max expected results (adjust as needed)
	var results [MAX_RESULTS]ProductResult
	count := 0

	for {
		msg := <-resultIn

		switch msg.Type {
		case MSG_ADD_RESULT:
			if count >= MAX_RESULTS {
				fmt.Println("[ResultThread] Warning: results full!")
				continue
			}

			result := msg.Result

			// Find insertion position
			insertPos := count
			for i := 0; i < count; i++ {
				if results[i].Product.Name > result.Product.Name {
					insertPos = i
					break
				}
			}

			// Shift elements
			for i := count; i > insertPos; i-- {
				results[i] = results[i-1]
			}

			results[insertPos] = result
			count++

		case MSG_GET_RESULTS:
			// Convert to slice for sending
			resultSlice := make([]ProductResult, count)
			for i := 0; i < count; i++ {
				resultSlice[i] = results[i]
			}

			resultOut <- Message{
				Type:    MSG_RESULTS_RESPONSE,
				Results: resultSlice,
			}
			return
		}
	}
}

// WorkerThread processes data items
func WorkerThread(id int, dataIn chan Message, dataOut chan Message,
	resultIn chan Message, doneSignal chan bool) {

	fmt.Printf("[Worker %d] Started\n", id)
	processed := 0

	for {
		dataIn <- Message{Type: MSG_REQUEST_DATA}
		response := <-dataOut

		switch response.Type {
		case MSG_NO_DATA:
			// Temporarily no data - retry
			time.Sleep(5 * time.Millisecond)
			continue

		case MSG_DONE:
			// Really finished - exit gracefully
			fmt.Printf("[Worker %d] Finished, processed %d items\n", id, processed)
			doneSignal <- true
			return

		case MSG_DATA_RESPONSE:
			product := response.Product
			riskScore := calculateRiskScore(product)

			if isHighRisk(riskScore) {
				resultIn <- Message{
					Type: MSG_ADD_RESULT,
					Result: ProductResult{
						Product:   product,
						RiskScore: riskScore,
					},
				}
			}

			processed++
			if processed%10 == 0 {
				fmt.Printf("[Worker %d] Processed %d items\n", id, processed)
			}
		}
	}
}

// writeResultsToFile writes results to output file
func writeResultsToFile(filename string, results []ProductResult) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	totalQuantity := 0
	totalPrice := 0.0
	for _, r := range results {
		totalQuantity += r.Product.Quantity
		totalPrice += r.Product.Price
	}

	fmt.Fprintf(file, "%-30s %-12s %-12s %-15s\n", "Name", "Quantity", "Price", "Risk Score")
	fmt.Fprintf(file, "%s\n", "---------------------------------------------------------------------")

	for _, r := range results {
		fmt.Fprintf(file, "%-30s %-12d %-12.2f %-15.5e\n",
			r.Product.Name, r.Product.Quantity, r.Product.Price, r.RiskScore)
	}

	fmt.Fprintf(file, "%s\n", "=====================================================================")
	fmt.Fprintf(file, "%-30s %-12d %-12.2f\n", "TOTALS (filtered items only):", totalQuantity, totalPrice)

	return nil
}

func main() {
	const inputFile = "IF31_RauluseviciusR_L1_dat_1.json"
	const outputFile = "IF31_RauluseviciusR_L2_rez.txt"

	totalStart := time.Now()

	fmt.Println("-- Loading products from file:", inputFile, "--")
	loadStart := time.Now()
	products, err := loadProductsFromFile(inputFile)
	if err != nil {
		fmt.Println("Error loading file:", err)
		return
	}
	loadDuration := time.Since(loadStart)
	fmt.Printf("Loaded %d products in %v\n", len(products), loadDuration)

	n := len(products)

	numWorkers := n / 4
	if numWorkers < 2 {
		numWorkers = 2
	}
	if numWorkers > 8 {
		numWorkers = 8
	}
	fmt.Printf("\n-- Using %d workers with Go channels --\n", numWorkers)

	// 1. Main → DataThread communication
	dataThreadIn := make(chan Message)
	// Main sends commands to DataThread
	// Workers send requests to DataThread

	// 2. DataThread → Main/Workers communication
	dataThreadOut := make(chan Message)
	// DataThread sends responses back
	// DataThread sends acknowledgments

	// 3. Workers → ResultThread communication
	resultThreadIn := make(chan Message)
	// Workers send their calculated results

	// 4. ResultThread → Main communication
	resultThreadOut := make(chan Message)
	// ResultThread sends all results back at the end

	// 5. Workers → Main completion signal
	workerDone := make(chan bool)
	// Workers signal when they're finished

	// Start DataThread and ResultThread
	go DataThread(dataThreadIn, dataThreadOut)
	go ResultThread(resultThreadIn, resultThreadOut)

	fmt.Println("\n-- Starting workers --")
	processStart := time.Now()

	for i := 0; i < numWorkers; i++ {
		go WorkerThread(i, dataThreadIn, dataThreadOut, resultThreadIn, workerDone)
	}

	// 5. Send all products to DataThread
	// Keep retrying until all items are successfully added
	fmt.Println("\n-- Sending data to DataThread --")
	itemsSent := 0
	for itemsSent < len(products) {
		// Try to send the next item
		time.Sleep(1 * time.Second)
		dataThreadIn <- Message{
			Type:    MSG_ADD_DATA,
			Product: products[itemsSent],
		}

		// Wait for acknowledgment
		response := <-dataThreadOut
		switch response.Type {
		case MSG_DATA_ADDED:
			itemsSent++
			if itemsSent%10 == 0 {
				fmt.Printf("Sent %d/%d items\n", itemsSent, n)
			}
		case MSG_DATA_FULL:
			// Array is full, wait a bit before retrying
			time.Sleep(1 * time.Millisecond)
		}
	}
	fmt.Printf("All %d items sent to DataThread\n", n)

	// Signal that producer is done
	dataThreadIn <- Message{Type: MSG_DONE}

	fmt.Println("\n-- Waiting for workers to complete --")
	for i := 0; i < numWorkers; i++ {
		<-workerDone
	}
	processDuration := time.Since(processStart)

	fmt.Println("\n-- Processing complete --")
	fmt.Printf("Processing time: %v\n", processDuration)

	fmt.Println("\n-- Requesting results --")
	resultThreadIn <- Message{Type: MSG_GET_RESULTS}
	resultMsg := <-resultThreadOut
	results := resultMsg.Results

	fmt.Printf("Results found: %d out of %d products\n", len(results), n)

	writeStart := time.Now()
	err = writeResultsToFile(outputFile, results)
	if err != nil {
		fmt.Println("Error writing results:", err)
		return
	}
	writeDuration := time.Since(writeStart)
	fmt.Println("Results written to", outputFile)

	dataThreadIn <- Message{Type: MSG_DONE}

	totalDuration := time.Since(totalStart)

	fmt.Println("\n-- Execution Summary --")
	fmt.Printf("Loading time:    %v\n", loadDuration)
	fmt.Printf("Processing time: %v\n", processDuration)
	fmt.Printf("Writing time:    %v\n", writeDuration)
	fmt.Printf("Total time:      %v\n", totalDuration)

	totalQuantity := 0
	totalPrice := 0.0
	for _, r := range results {
		totalQuantity += r.Product.Quantity
		totalPrice += r.Product.Price
	}

	fmt.Println("\n-- Sums (filtered items only) --")
	fmt.Printf("Total Quantity: %d\n", totalQuantity)
	fmt.Printf("Total Price: %.2f\n", totalPrice)
	fmt.Println("\n-- Done. --")
}
