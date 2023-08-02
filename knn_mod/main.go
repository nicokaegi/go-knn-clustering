package main

import "fmt"
import "os" 
import "log"
import "encoding/csv"
import "strconv"
import "math"
import "sort"
import "time"

type Compare func(slice1 []int, slice2 []int) float64

func main(){

    var data_set [][]int

    train_path := "mnist_small_knn/train.csv"

    data_set = load_csv(train_path)
	
    test_path := "mnist_small_knn/test.csv"

    test_set := load_csv(test_path)
    first := time.Now()

    k_range := []int {1, 3, 5, 10, 15}
    fmt.Println(k_range)

    for _, k := range k_range {

        first = time.Now()
        classifications := classifiy_k_nearst(1, test_set, data_set, Euclidian_distance)
        fmt.Println(" Euclidian_distance : " , "k : ", k,"accuracy : ", eval(test_set, classifications), "time : ", time.Now().Sub(first))

        first = time.Now()
        classifications = classifiy_k_nearst(1, test_set, data_set, Manhattan_distance)
        fmt.Println(" Manhattan_distance : ", "k : ", k,"accuracy : ", eval(test_set, classifications), "time : ", time.Now().Sub(first))

    }

}

func eval(target_set [][]int, classifications []int) float64{

    /*
        input : the orginal test set, and a list of proposed classifications

        ouput : a float that represents the accuracy of the knn

    */

    right := 0.0

    for target_pos := 0; target_pos < len(target_set); target_pos++{
        
        if target_set[target_pos][0] == classifications[target_pos] {

            right++

        }
    
    }

    return right/float64(len(target_set))

}

func classifiy_k_nearst(k int, target_set [][]int, data_set [][]int, compare Compare) []int{

    /*
        input : the k in knn, the set of reconrds to be classified, the orginal training set, the function that you use to judge similarty 

        output : a set of classifications
    */

    output_slice := make([]int, len(target_set))

    for target_pos := 0; target_pos < len(target_set); target_pos++{
        distances := make([][]float64, len(data_set))

        var target_slice []int

        target_slice = target_set[target_pos][1:]

        for pos := 0; pos < len(data_set); pos++{

            distances[pos] = []float64 { compare(target_slice, data_set[pos][1:]),  float64(data_set[pos][0]) }
        }

        sort.Slice(distances, func(i, j int) bool { 
            return distances[i][0] < distances[j][0]
        })

        top_k := distances[:k]

        class_count := make(map[float64]int, 10)
        for pos := 0; pos < k; pos++{
            class_count[top_k[pos][1]] += 1

        }

        largest_count := -math.MaxInt
        output := -1.0
        for k, v := range class_count {
            if (largest_count < v ){
                largest_count = v
                output = k 
            }
        }

        output_slice[target_pos] = int(output)
    }
    
    return output_slice

}

func Euclidian_distance(slice1 []int, slice2 []int) float64{


    temp_float := 0.0

    for pos := 0; pos < len(slice1) ;pos++{

        temp_float += math.Pow( (float64(slice1[pos]) - float64(slice2[pos])), 2.0)

    }

    return math.Sqrt(temp_float)

}


func Manhattan_distance(slice1 []int, slice2 []int) float64{

    temp_float := 0.0

    for pos := 0; pos < len(slice1) ;pos++{

        temp_float += math.Abs(float64(slice1[pos]) - float64(slice2[pos]))

    }

    return temp_float

}

func load_csv(csv_path string) [][]int{

    /*
        input : path to data csv
        
        output : input csv represetned as a multidimsional list of ints

    */

    csv_f, err := os.Open(csv_path)
    if err != nil {
        log.Fatal(err)
    }

    defer csv_f.Close()

    r := csv.NewReader(csv_f)
    var intial_data, _ = r.ReadAll()
    data_set := make([][]int, len(intial_data))
	for pos := 0; pos < len(intial_data); pos++{

        data_set[pos] = string_list_int_list(intial_data[pos])

	}

    return data_set

}


func string_list_int_list(string_list []string) []int{
    
    /*
        input : list of strings
        
        output : a brand spanking new list of ints only for 3 paymentts of 9,99

    */

    var out_list []int 

    for _, i := range string_list {
        j, err := strconv.Atoi(i)
        if err != nil {
            panic(err)
        }
        out_list = append(out_list, j)
    }

    return out_list

}
