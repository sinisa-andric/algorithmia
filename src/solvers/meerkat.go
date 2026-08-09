package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	meerkatPopulation    = 20
	meerkatMaxSteps      = 1000
	meerkatTolerance     = 1e-6
	meerkatRangeLow      = -5.0
	meerkatRangeHigh     = 5.0
	meerkatSentinelRatio = 0.2
)

// Meerkat minimizuje konfigurisanu benchmark funkciju koristeći Meerkat Optimization Algorithm: deo populacije
// (stražari) vrši malu lokalnu pretragu oko najboljeg rešenja, dok ostali (foraging) traže hranu krećući se ka
// najboljem rešenju uz uticaj nasumičnog člana jata
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func Meerkat(problem models.Problem) (result models.Result, err error) {

	if len(problem.Point) == 0 {
		err = fmt.Errorf("starting point is required")
		return result, err
	}

	fnName, _ := problem.Payload["function"].(string)
	fn, err := functions.Get(fnName)
	if err != nil {
		return result, err
	}
	if err := fn.ValidateDimension(len(problem.Point)); err != nil {
		return result, err
	}

	population := meerkatPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := meerkatMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := meerkatTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	sentinelRatio := meerkatSentinelRatio
	if v, ok := problem.Payload["sentinel_ratio"].(float64); ok {
		sentinelRatio = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	sortByValue := func(points [][]float64, vals []float64) {
		idx := make([]int, len(points))
		for i := range idx {
			idx[i] = i
		}
		for i := 1; i < len(idx); i++ {
			for j := i; j > 0 && vals[idx[j]] < vals[idx[j-1]]; j-- {
				idx[j], idx[j-1] = idx[j-1], idx[j]
			}
		}
		sortedPoints := make([][]float64, len(points))
		sortedVals := make([]float64, len(points))
		for i, k := range idx {
			sortedPoints[i] = points[k]
			sortedVals[i] = vals[k]
		}
		copy(points, sortedPoints)
		copy(vals, sortedVals)
	}

	randomIndexExcept := func(exclude int) int {
		if population < 2 {
			return exclude
		}
		j := rand.IntN(population)
		for j == exclude {
			j = rand.IntN(population)
		}
		return j
	}

	meerkats := make([][]float64, population)
	values := make([]float64, population)
	for i := range meerkats {
		meerkat := make([]float64, dimensions)
		for d := range meerkat {
			meerkat[d] = meerkatRangeLow + rand.Float64()*(meerkatRangeHigh-meerkatRangeLow)
		}
		meerkats[i] = meerkat
		values[i] = fn.Evaluate(meerkat)
	}

	best := append([]float64(nil), meerkats[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), meerkats[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		sortByValue(meerkats, values)

		numSentinels := max(0, min(population, int(math.Round(sentinelRatio*float64(population)))))

		for i := range numSentinels {
			for d := range meerkats[i] {
				meerkats[i][d] = best[d] + (rand.Float64()*2-1)*0.2
				meerkats[i][d] = clamp(meerkats[i][d], meerkatRangeLow, meerkatRangeHigh)
			}
			values[i] = fn.Evaluate(meerkats[i])
		}

		for i := numSentinels; i < population; i++ {
			r := meerkats[randomIndexExcept(i)]
			for d := range meerkats[i] {
				meerkats[i][d] = meerkats[i][d] + rand.Float64()*(best[d]-meerkats[i][d]) + rand.Float64()*(r[d]-meerkats[i][d])*0.3
				meerkats[i][d] = clamp(meerkats[i][d], meerkatRangeLow, meerkatRangeHigh)
			}
			values[i] = fn.Evaluate(meerkats[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), meerkats[i]...)
			}
			if includeTrajectory {
				trajectory = recordTrajectory(trajectory, steps, best, bestValue, false)
			}
		}

		if math.Abs(bestValue-prevBestValue) < tolerance {
			noImprove++
		} else {
			noImprove = 0
		}
		if noImprove >= maxNoImprove {
			steps++
			break
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, best, bestValue, true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "meerkat",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
