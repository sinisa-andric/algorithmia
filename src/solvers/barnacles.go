package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	barnaclesPopulation = 20
	barnaclesMaxSteps   = 1000
	barnaclesTolerance  = 1e-6
	barnaclesRangeLow   = -5.0
	barnaclesRangeHigh  = 5.0
)

// Barnacles minimizuje konfigurisanu benchmark funkciju koristeći Barnacles Mating Optimizer: svaki barnakl se,
// sa jednakom verovatnoćom, ili pari nasumičnim izlivanjem sperme ka drugom barnaklu (istraživanje) ili leže ka
// najboljem rešenju (eksploatacija)
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func Barnacles(problem models.Problem) (result models.Result, err error) {

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

	population := barnaclesPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := barnaclesMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := barnaclesTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
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

	barnacles := make([][]float64, population)
	values := make([]float64, population)
	for i := range barnacles {
		barnacle := make([]float64, dimensions)
		for d := range barnacle {
			barnacle[d] = barnaclesRangeLow + rand.Float64()*(barnaclesRangeHigh-barnaclesRangeLow)
		}
		barnacles[i] = barnacle
		values[i] = fn.Evaluate(barnacle)
	}

	best := append([]float64(nil), barnacles[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), barnacles[i]...)
		}
	}

	const p = 0.5

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range barnacles {
			if rand.Float64() < p {
				r := barnacles[randomIndexExcept(i)]
				for d := range barnacles[i] {
					barnacles[i][d] = barnacles[i][d] + rand.Float64()*(r[d]-barnacles[i][d])
				}
			} else {
				for d := range barnacles[i] {
					barnacles[i][d] = barnacles[i][d] + rand.Float64()*(best[d]-barnacles[i][d])
				}
			}

			for d := range barnacles[i] {
				barnacles[i][d] = clamp(barnacles[i][d], barnaclesRangeLow, barnaclesRangeHigh)
			}
			values[i] = fn.Evaluate(barnacles[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), barnacles[i]...)
			}
		}

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
		Method:     "barnacles",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
