package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	shuffledFrogPopulation = 20
	shuffledFrogMaxSteps   = 1000
	shuffledFrogTolerance  = 1e-6
	shuffledFrogRangeLow   = -5.0
	shuffledFrogRangeHigh  = 5.0
	shuffledFrogMemeplexes = 3
)

// ShuffledFrog minimizuje konfigurisanu benchmark funkciju koristeći Shuffled Frog Leaping Algorithm: populacija se
// sortira i deli po modulu u memeplexe, u svakom memeplexu se najgora žaba pomera prema najboljoj iz istog memeplexa,
// uz fallback na globalno najbolje rešenje
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func ShuffledFrog(problem models.Problem) (result models.Result, err error) {

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

	population := shuffledFrogPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := shuffledFrogMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := shuffledFrogTolerance
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

	frogs := make([][]float64, population)
	values := make([]float64, population)
	for i := range frogs {
		frog := make([]float64, dimensions)
		for d := range frog {
			frog[d] = shuffledFrogRangeLow + rand.Float64()*(shuffledFrogRangeHigh-shuffledFrogRangeLow)
		}
		frogs[i] = frog
		values[i] = fn.Evaluate(frog)
	}
	sortByValue(frogs, values)

	best := append([]float64(nil), frogs[0]...)
	bestValue := values[0]

	numMemeplexes := min(shuffledFrogMemeplexes, population)

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		sortByValue(frogs, values)

		memeplexIdx := make([][]int, numMemeplexes)
		for i := 0; i < population; i++ {
			m := i % numMemeplexes
			memeplexIdx[m] = append(memeplexIdx[m], i)
		}

		for _, idxs := range memeplexIdx {
			if len(idxs) == 0 {
				continue
			}
			localBestIdx := idxs[0]
			localWorstIdx := idxs[len(idxs)-1]
			localBest := frogs[localBestIdx]
			localWorst := frogs[localWorstIdx]

			novi := make([]float64, dimensions)
			for d := range novi {
				dVal := rand.Float64() * (localBest[d] - localWorst[d])
				novi[d] = clamp(localWorst[d]+dVal, shuffledFrogRangeLow, shuffledFrogRangeHigh)
			}
			noviValue := fn.Evaluate(novi)

			if noviValue < values[localWorstIdx] {
				frogs[localWorstIdx] = novi
				values[localWorstIdx] = noviValue
			} else {
				for d := range frogs[localWorstIdx] {
					frogs[localWorstIdx][d] = clamp(best[d]+rand.Float64()*(best[d]-frogs[localWorstIdx][d]), shuffledFrogRangeLow, shuffledFrogRangeHigh)
				}
				values[localWorstIdx] = fn.Evaluate(frogs[localWorstIdx])
			}
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), frogs[i]...)
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
		Method:     "shuffled_frog",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
