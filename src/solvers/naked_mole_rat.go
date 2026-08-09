package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	nakedMoleRatPopulation = 20
	nakedMoleRatMaxSteps   = 1000
	nakedMoleRatTolerance  = 1e-6
	nakedMoleRatRangeLow   = -5.0
	nakedMoleRatRangeHigh  = 5.0
)

// NakedMoleRat minimizuje konfigurisanu benchmark funkciju koristeći Naked Mole-Rat Algorithm: radnici najčešće prate
// kraljicu (globalno najbolje rešenje), a povremeno nasumično istražuju prostor pretrage prateći razliku u odnosu na
// drugog radnika
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func NakedMoleRat(problem models.Problem) (result models.Result, err error) {

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

	population := nakedMoleRatPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := nakedMoleRatMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := nakedMoleRatTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
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

	moleRats := make([][]float64, population)
	values := make([]float64, population)
	for i := range moleRats {
		moleRat := make([]float64, dimensions)
		for d := range moleRat {
			moleRat[d] = nakedMoleRatRangeLow + rand.Float64()*(nakedMoleRatRangeHigh-nakedMoleRatRangeLow)
		}
		moleRats[i] = moleRat
		values[i] = fn.Evaluate(moleRat)
	}
	sortByValue(moleRats, values)

	queen := append([]float64(nil), moleRats[0]...)
	queenValue := values[0]
	workers := moleRats[1:]
	workerValues := values[1:]

	randomIndexExcept := func(n, exclude int) int {
		if n < 2 {
			return exclude
		}
		j := rand.IntN(n)
		for j == exclude {
			j = rand.IntN(n)
		}
		return j
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := queenValue

		for i := range workers {
			if rand.Float64() < 0.7 {
				for d := range workers[i] {
					workers[i][d] = workers[i][d] + rand.Float64()*(queen[d]-workers[i][d])
				}
			} else {
				r := workers[randomIndexExcept(len(workers), i)]
				for d := range workers[i] {
					workers[i][d] = workers[i][d] + rand.Float64()*(workers[i][d]-r[d])
				}
			}

			for d := range workers[i] {
				workers[i][d] = clamp(workers[i][d], nakedMoleRatRangeLow, nakedMoleRatRangeHigh)
			}
			workerValues[i] = fn.Evaluate(workers[i])
		}

		for i, v := range workerValues {
			if v < queenValue {
				queenValue = v
				queen = append([]float64(nil), workers[i]...)
			}
		}

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, queen, queenValue, false)
		}

		if math.Abs(queenValue-prevBestValue) < tolerance {
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
		trajectory = recordTrajectory(trajectory, steps, queen, queenValue, true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "naked_mole_rat",
		Point:      queen,
		Value:      queenValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
