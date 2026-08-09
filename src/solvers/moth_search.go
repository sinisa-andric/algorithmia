package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	mothSearchPopulation = 20
	mothSearchMaxSteps   = 1000
	mothSearchTolerance  = 1e-6
	mothSearchRangeLow   = -5.0
	mothSearchRangeHigh  = 5.0
	mothSearchMaxRatio   = 0.8
	mothSearchScale      = 1.0
)

// MothSearch minimizuje konfigurisanu benchmark funkciju koristeći Moth Search Algorithm: fototaktička grupa moljaca
// (deo populacije određen max_ratio) koristi Levy let za istraživanje, dok ostali moljci koriste sve precizniju
// korekciju ka najboljem rešenju tokom izvršavanja
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func MothSearch(problem models.Problem) (result models.Result, err error) {

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

	population := mothSearchPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := mothSearchMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := mothSearchTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	maxRatio := mothSearchMaxRatio
	if v, ok := problem.Payload["max_ratio"].(float64); ok {
		maxRatio = v
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

	moths := make([][]float64, population)
	values := make([]float64, population)
	for i := range moths {
		moth := make([]float64, dimensions)
		for d := range moth {
			moth[d] = mothSearchRangeLow + rand.Float64()*(mothSearchRangeHigh-mothSearchRangeLow)
		}
		moths[i] = moth
		values[i] = fn.Evaluate(moth)
	}

	best := append([]float64(nil), moths[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), moths[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		sortByValue(moths, values)

		numMoths1 := int(math.Round(maxRatio * float64(population)))
		numMoths1 = max(0, min(numMoths1, population))

		for i := 0; i < numMoths1; i++ {
			for d := range moths[i] {
				l := mantegnaLevy(1.5)
				moths[i][d] = moths[i][d] + mothSearchScale*l*(moths[i][d]-best[d])
				moths[i][d] = clamp(moths[i][d], mothSearchRangeLow, mothSearchRangeHigh)
			}
			values[i] = fn.Evaluate(moths[i])
		}

		for i := numMoths1; i < population; i++ {
			for d := range moths[i] {
				moths[i][d] = moths[i][d] + rand.Float64()*(best[d]-moths[i][d])/float64(steps+1)
				moths[i][d] = clamp(moths[i][d], mothSearchRangeLow, mothSearchRangeHigh)
			}
			values[i] = fn.Evaluate(moths[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), moths[i]...)
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
		Method:     "moth_search",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
