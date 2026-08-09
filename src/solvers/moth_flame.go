package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	mothFlamePopulation = 20
	mothFlameMaxSteps   = 1000
	mothFlameTolerance  = 1e-6
	mothFlameRangeLow   = -5.0
	mothFlameRangeHigh  = 5.0
)

// MothFlame minimizuje konfigurisanu benchmark funkciju koristeći Moth-Flame Optimization: svaki moljac se kreće
// spiralno oko svog dodeljenog plamena, a plamenovi se nakon svakog koraka ažuriraju sortiranjem trenutne populacije
// po vrednosti funkcije
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func MothFlame(problem models.Problem) (result models.Result, err error) {

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

	population := mothFlamePopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := mothFlameMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := mothFlameTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	moths := make([][]float64, population)
	values := make([]float64, population)
	for i := range moths {
		moth := make([]float64, dimensions)
		for d := range moth {
			moth[d] = mothFlameRangeLow + rand.Float64()*(mothFlameRangeHigh-mothFlameRangeLow)
		}
		moths[i] = moth
		values[i] = fn.Evaluate(moth)
	}

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

	flames := make([][]float64, population)
	flameValues := make([]float64, population)
	for i := range moths {
		flames[i] = append([]float64(nil), moths[i]...)
		flameValues[i] = values[i]
	}
	sortByValue(flames, flameValues)

	bestValue := flameValues[0]
	best := append([]float64(nil), flames[0]...)

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		numFlames := max(1, int(math.Round(float64(population)-float64(steps)*float64(population-1)/float64(maxSteps))))

		for i := range moths {
			flameIdx := min(i, numFlames-1)
			for d := range moths[i] {
				dist := math.Abs(flames[flameIdx][d] - moths[i][d])
				b := 1.0
				t := rand.Float64() - 1
				moths[i][d] = dist*math.Exp(b*t)*math.Cos(2*math.Pi*t) + flames[flameIdx][d]
				moths[i][d] = clamp(moths[i][d], mothFlameRangeLow, mothFlameRangeHigh)
			}
			values[i] = fn.Evaluate(moths[i])
		}

		for i := range moths {
			flames[i] = append([]float64(nil), moths[i]...)
			flameValues[i] = values[i]
		}
		sortByValue(flames, flameValues)

		bestValue = flameValues[0]
		best = append([]float64(nil), flames[0]...)

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, best, bestValue, false)
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
		Method:     "moth_flame",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
