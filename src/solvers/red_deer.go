package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	redDeerPopulation = 20
	redDeerMaxSteps   = 1000
	redDeerTolerance  = 1e-6
	redDeerRangeLow   = -5.0
	redDeerRangeHigh  = 5.0
)

// RedDeer minimizuje konfigurisanu benchmark funkciju koristeći Red Deer Algorithm: populacija se deli na rogonoše
// (bolja trećina) i košute, rogonoše se takmiče međusobno i ka najboljem rešenju,
// dok košute prate nasumičnog komandanta iz harema rogonoša
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func RedDeer(problem models.Problem) (result models.Result, err error) {

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

	population := redDeerPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := redDeerMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := redDeerTolerance
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

	deer := make([][]float64, population)
	values := make([]float64, population)
	for i := range deer {
		d := make([]float64, dimensions)
		for k := range d {
			d[k] = redDeerRangeLow + rand.Float64()*(redDeerRangeHigh-redDeerRangeLow)
		}
		deer[i] = d
		values[i] = fn.Evaluate(d)
	}
	sortByValue(deer, values)

	numMales := max(1, (2*population)/3)
	if numMales >= population {
		numMales = population - 1
	}
	if numMales < 1 {
		numMales = 1
	}

	males := deer[:numMales]
	hinds := deer[numMales:]

	best := append([]float64(nil), deer[0]...)
	bestValue := values[0]

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

		prevBestValue := bestValue

		for i := range males {
			otherIdx := randomIndexExcept(len(males), i)
			other := males[otherIdx]
			for d := range males[i] {
				males[i][d] = males[i][d] + rand.Float64()*(best[d]-males[i][d]) + rand.Float64()*(males[i][d]-other[d])
				males[i][d] = clamp(males[i][d], redDeerRangeLow, redDeerRangeHigh)
			}
		}

		for i := range hinds {
			commander := males[rand.IntN(len(males))]
			for d := range hinds[i] {
				hinds[i][d] = hinds[i][d] + rand.Float64()*(commander[d]-hinds[i][d])
				hinds[i][d] = clamp(hinds[i][d], redDeerRangeLow, redDeerRangeHigh)
			}
		}

		for i := range deer {
			values[i] = fn.Evaluate(deer[i])
		}
		sortByValue(deer, values)
		males = deer[:numMales]
		hinds = deer[numMales:]

		bestValue = values[0]
		best = append([]float64(nil), deer[0]...)

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
		Method:     "red_deer",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
