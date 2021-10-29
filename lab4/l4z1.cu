#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>
#include <thrust/count.h>
// zadania 1-5 https://e.mini.pw.edu.pl/en/node/7130

#define N 10

struct Data {
    thrust::device_vector<int> day, site, measurement;

    Data() : day(N), site(N), measurement(N) {
        int day_data[] = {0, 0, 1, 2, 5, 5, 6, 6, 7, 8};
        int site_data[] = {2, 3, 0, 1, 1, 2, 0, 1, 2, 1};
        int measurement_data[] = {9, 5, 6, 3, 3, 8, 2, 6, 5, 10};
        thrust::copy(day_data, day_data + N, day.begin());
        thrust::copy(site_data, site_data + N, site.begin());
        thrust::copy(measurement_data, measurement_data + N, measurement.begin());
    }
};

// #2: 
void sum_in_groups(const Data& data, thrust::device_vector<int>& groups, thrust::device_vector<int>& values) {
    thrust::device_vector<int> local_site(data.site),
        local_measurement(data.measurement);
    thrust::sort_by_key(local_site.begin(), local_site.end(), local_measurement.begin());
    thrust::reduce_by_key(local_site.begin(), local_site.end(), local_measurement.begin(), groups.begin(), values.begin());
}

// #4: 
struct Above_5
{
  __host__ __device__
  bool operator()(const int &x)
  {
    return x > 5;
  }
};

// #5:
void check_groups(const Data& data, thrust::device_vector<int>& groups, thrust::device_vector<bool>& values) {
    thrust::device_vector<int> local_day(data.day),
        local_measurement(data.measurement);
    thrust::sort_by_key(local_day.begin(), local_day.end(), local_measurement.begin());
    thrust::reduce_by_key(local_site.begin(), local_site.end(), local_measurement.begin(), groups.begin(), values.begin(), thrust::binary_partition);
}

int main(int argc, char **argv) {
    Data data = Data();
    std::cout << data.day[0] << std::endl;

    // #1: inner_product: https://thrust.github.io/doc/group__transformed__reductions_gad9df36f7648745ca572037727b66b48d.html#gad9df36f7648745ca572037727b66b48d
    int num_days_with_rainfall = thrust::inner_product(data.day.begin(), data.day.end() - 1, data.day.begin() + 1, 1, thrust::plus<int>(), thrust::not_equal_to<int>());
    std::cout << "num_days_with_rainfall:" << num_days_with_rainfall << std::endl;

    // #2: reduce_by_key: https://thrust.github.io/doc/group__reductions_gad5623f203f9b3fdcab72481c3913f0e0.html, tylko trzeba miec posortowane dane: https://thrust.github.io/doc/group__sorting_ga2bb765aeef19f6a04ca8b8ba11efff24.html#ga2bb765aeef19f6a04ca8b8ba11efff24
    thrust::device_vector<int> groups(N), values(N);
    sum_in_groups(data, groups, values);

    // #3:
    for (int i = 0; i < groups.size(); ++i) {
        std::cout << groups[i] << ": " << values[i] << " | ";
    }
    std::cout << std::endl;

    // #4: count_if: https://thrust.github.io/doc/group__counting_gae2f8874093d33f3f0f49b51d8b26438c.html#gae2f8874093d33f3f0f49b51d8b26438c
    //
    int num_of_days_with_rain_above_5 = thrust::count_if(data.measurement.begin(), data.measurement.end(), Above_5());
    std::cout << "num_of_days_with_rain_above_5:" << num_of_days_with_rain_above_5 << std::endl;

    // #5:
    thrust::transform 	( 	const thrust::detail::execution_policy_base< DerivedPolicy > &  	exec,
		InputIterator  	first,
		InputIterator  	last,
		OutputIterator  	result,
		UnaryFunction  	op 
	) 	


    return 0;
}