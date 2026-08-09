package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	nutcrackerPopulation = 20
	nutcrackerMaxSteps   = 1000
	nutcrackerTolerance  = 1e-6
	nutcrackerRangeLow   = -5.0
	nutcrackerRangeHigh  = 5.0
	nutcrackerPA         = 0.2
)

// Nutcracker minimizuje konfigurisanu benchmark funkciju koristeći Nutcracker Optimizer: kreker semenki ili skladišti
// hranu Levy letom ka najboljem rešenju ili traži zaboravljeno skladište krećući se ka nasumičnoj zapamćenoj (cache)
// poziciji
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func Nutcracker(problem models.Problem) (result models.Result, err error) {

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

	population := nutcrackerPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := nutcrackerMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := nutcrackerTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	pa := nutcrackerPA
	if v, ok := problem.Payload["pa"].(float64); ok {
		pa = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

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

	nutcrackers := make([][]float64, population)
	values := make([]float64, population)
	for i := range nutcrackers {
		nutcracker := make([]float64, dimensions)
		for d := range nutcracker {
			nutcracker[d] = nutcrackerRangeLow + rand.Float64()*(nutcrackerRangeHigh-nutcrackerRangeLow)
		}
		nutcrackers[i] = nutcracker
		values[i] = fn.Evaluate(nutcracker)
	}

	cache := make([][]float64, population)
	cacheValues := make([]float64, population)
	for i := range nutcrackers {
		cache[i] = append([]float64(nil), nutcrackers[i]...)
		cacheValues[i] = values[i]
	}

	best := append([]float64(nil), nutcrackers[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), nutcrackers[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range nutcrackers {
			if rand.Float64() < pa {
				l := mantegnaLevy(1.5)
				for d := range nutcrackers[i] {
					nutcrackers[i][d] = nutcrackers[i][d] + l*(best[d]-nutcrackers[i][d])
				}
			} else {
				r := cache[randomIndexExcept(i)]
				for d := range nutcrackers[i] {
					nutcrackers[i][d] = nutcrackers[i][d] + rand.Float64()*(r[d]-nutcrackers[i][d])
				}
			}

			for d := range nutcrackers[i] {
				nutcrackers[i][d] = clamp(nutcrackers[i][d], nutcrackerRangeLow, nutcrackerRangeHigh)
			}
			values[i] = fn.Evaluate(nutcrackers[i])

			if values[i] < cacheValues[i] {
				cache[i] = append([]float64(nil), nutcrackers[i]...)
				cacheValues[i] = values[i]
			}
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), nutcrackers[i]...)
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
		Method:     "nutcracker",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
