package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	swallowPopulation = 20
	swallowMaxSteps   = 1000
	swallowTolerance  = 1e-6
	swallowRangeLow   = -5.0
	swallowRangeHigh  = 5.0
)

// Swallow minimizuje konfigurisanu benchmark funkciju koristeći Swallow Swarm Optimization: populacija se svakog
// koraka sortira i deli na vođu (najbolji), istraživače (60% populacije, koji prate vođu inercijom brzine) i one bez
// cilja (ostatak, koji nasumično lutaju)
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func Swallow(problem models.Problem) (result models.Result, err error) {

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

	population := swallowPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := swallowMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := swallowTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	sortAll := func(points [][]float64, vals []float64, vel [][]float64) {
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
		sortedVel := make([][]float64, len(points))
		for i, k := range idx {
			sortedPoints[i] = points[k]
			sortedVals[i] = vals[k]
			sortedVel[i] = vel[k]
		}
		copy(points, sortedPoints)
		copy(vals, sortedVals)
		copy(vel, sortedVel)
	}

	swallows := make([][]float64, population)
	velocity := make([][]float64, population)
	values := make([]float64, population)
	for i := range swallows {
		swallow := make([]float64, dimensions)
		for d := range swallow {
			swallow[d] = swallowRangeLow + rand.Float64()*(swallowRangeHigh-swallowRangeLow)
		}
		swallows[i] = swallow
		velocity[i] = make([]float64, dimensions)
		values[i] = fn.Evaluate(swallow)
	}

	sortAll(swallows, values, velocity)

	best := append([]float64(nil), swallows[0]...)
	bestValue := values[0]

	explorerCount := max(1, min(population-1, int(math.Round(0.6*float64(population)))))

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		leader := swallows[0]

		for i := 1; i <= explorerCount && i < population; i++ {
			for d := range swallows[i] {
				velocity[i][d] = 0.7*velocity[i][d] + rand.Float64()*(leader[d]-swallows[i][d])
				swallows[i][d] = swallows[i][d] + velocity[i][d]
				swallows[i][d] = clamp(swallows[i][d], swallowRangeLow, swallowRangeHigh)
			}
			values[i] = fn.Evaluate(swallows[i])
		}

		for i := explorerCount + 1; i < population; i++ {
			for d := range swallows[i] {
				swallows[i][d] = swallows[i][d] + (rand.Float64()*2-1)*0.5
				swallows[i][d] = clamp(swallows[i][d], swallowRangeLow, swallowRangeHigh)
			}
			values[i] = fn.Evaluate(swallows[i])
		}

		sortAll(swallows, values, velocity)

		bestValue = values[0]
		best = append([]float64(nil), swallows[0]...)

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
		Method:     "swallow",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
