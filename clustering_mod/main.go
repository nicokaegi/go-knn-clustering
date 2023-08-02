package main

import "fmt"
import "os" 
import "log"
import "encoding/csv"
import "strconv"
import "math"
import "sort"
import "math/rand"

type Compare func(slice1 []int, slice2 []int) float64

func main(){

    var data_set [][]int

    train_path := "mnist_small_knn/train.csv"

    data_set = load_csv(train_path)

    cluster_values_k := []int{5, 7, 9, 10, 12, 15}
	
    for k := range cluster_values_k {

        fmt.Println("k : ", cluster_values_k[k], " Dunn, and davis Bouldin index : ", eval_k_means(cluster_k_means(cluster_values_k[k], data_set)))
    }
    fmt.Println("job done")

}

func max_inter_custer_distance(cluster [][]int) float64{


    largest_dist := 0.0
    var temp_dist float64
    for i := range cluster{
        
        for j := range cluster{

            if i != j {

                temp_dist = Euclidian_distance(cluster[i], cluster[j])

                if largest_dist < temp_dist{

                    largest_dist = temp_dist

                }
            }
        }
    }

    return largest_dist
}


func single_linkage(cluster1 [][]int, cluster2 [][]int) float64{

    var temp_dist float64
    mindist := math.MaxFloat64
    for i := range cluster1{

        for j := range cluster2{

            temp_dist = Euclidian_distance(cluster1[i],cluster2[j])

            if temp_dist < mindist {

                mindist = temp_dist

            }
        }
    }

    return mindist

}


func dunn_index(clusters [][][]int) float64{

    /* do Dunn index */
    
    /*  

        for each cluster get the linkages between all the other clusters, and its diamiter 

        get the max diamater 

        get the smallest linkeage 

        return  smallest linkeage / max diamater
    
    */

    //get the max diamater 

    max_cluster_dim := 0.0
    var temp_dist float64
    for i := range clusters{

        temp_dist = max_inter_custer_distance(clusters[i])

        if max_cluster_dim < temp_dist{

            max_cluster_dim = temp_dist

        }
    }

    //get the smallest linkeage 

    min_cluster_link := math.MaxFloat64
    for i := range clusters{

        for j := range clusters{

            if i != j{

                temp_dist = single_linkage(clusters[i], clusters[j])

                if temp_dist < min_cluster_link{

                    min_cluster_link = temp_dist
        
                }
            }
        }
    }

    return min_cluster_link/max_cluster_dim
}

func mean_distance_to_centroid(centroid []int, cul_points [][]int) float64{

    /*get the d ( ͡° ͜ʖ ͡°)*/

    var d_sum float64
    for i := range cul_points{

        d_sum += Euclidian_distance(centroid, cul_points[i])

    }


    return d_sum/float64(len(cul_points))

}

func davis_bouldin_index(clusters [][][]int) float64{

    var centroid_slice [][]int
    var d_slice []float64
    for i := range clusters{

        centroid_slice = append(centroid_slice, calculate_centroid(clusters[i]))
        d_slice = append(d_slice, mean_distance_to_centroid(centroid_slice[i], clusters[i]))

    }


    var largest_R float64
    var temp_R float64
    max_r_slice := []float64{}
    for i := range clusters{

        largest_R = 0.0
        for j := range clusters{
            
            if i != j{

                temp_R = (d_slice[i] + d_slice[j])/Euclidian_distance(centroid_slice[i],centroid_slice[j])

                if largest_R < temp_R{

                    largest_R = temp_R

                }
                

            }
        }

        max_r_slice = append(max_r_slice, largest_R)
    }

    r_sum := 0.0
    for i := range max_r_slice{

        r_sum += max_r_slice[i]

    }

    return r_sum/float64(len(max_r_slice))
}

func eval_k_means(clusters [][][]int) []float64{

    /* the first one is dunn the second davis Bouldin */

    return []float64{dunn_index(clusters), davis_bouldin_index(clusters)}

}

func eval_knn(target_set [][]int, classifications []int) float64{

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

func calculate_centroid(cluster [][]int) []int{

    output := make([]int, len(cluster[0])-1)
    temp_sum := 0
    for i := range output{

        temp_sum = 0
        for j := range cluster{

            temp_sum += cluster[j][i]

        }

        output[i] = temp_sum/len(cluster)

    }

    return output[1:]

}


func cluster_k_means(k int, dataset [][]int) [][][]int {

    for i := range dataset {
        j := rand.Intn(i + 1)
        dataset[i], dataset[j] = dataset[j], dataset[i]
    }

    k_clusters := make([][][]int, k)
    intial_k := dataset[:k]
    k_centroids := make([][]int, k)
    for i := range intial_k{

        k_clusters[i] = [][]int{intial_k[i]}
        k_centroids[i] = calculate_centroid([][]int{intial_k[i]})

    } 
    
    /*

        for record in dataset 

            compare against the k centroids and get the closest 

                place record in the kluster of the nearest centroid

                    recaulate that nearest centroid 
                    
    */

    rest_of_the_dataset := dataset[k:]
    var smallest_dist float64
    var temp_dist float64
    var nearest_cluster int
    for i := range rest_of_the_dataset{

        smallest_dist = math.MaxFloat64
        nearest_cluster = -1
        for j := range k_centroids {

            temp_dist = Euclidian_distance(k_centroids[j], rest_of_the_dataset[i])
            if ( temp_dist < smallest_dist){

                smallest_dist = temp_dist
                nearest_cluster = j

            }
        }

        k_clusters[nearest_cluster] = append(k_clusters[nearest_cluster] ,  rest_of_the_dataset[i])
        k_centroids[nearest_cluster] = calculate_centroid(k_clusters[nearest_cluster])

    } 

    return k_clusters

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