package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	crestedPorcupinePopulation = 20
	crestedPorcupineMaxSteps   = 1000
	crestedPorcupineTolerance  = 1e-6
	crestedPorcupineRangeLow   = -5.0
	crestedPorcupineRangeHigh  = 5.0
)

// CrestedPorcupine minimizuje konfigurisanu benchmark funkciju koristeći Crested Porcupine Optimizer: bodljikavi
// prasac nasumično bira jedan od četiri mehanizma odbrane — vizuelni (ka najboljem), zvučni (nasumično lutanje),
// mirisni (odbojnost koja opada tokom izvršavanja) ili fizički napad
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func CrestedPorcupine(problem models.Problem) (result models.Result, err error) {

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

	population := crestedPorcupinePopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := crestedPorcupineMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := crestedPorcupineTolerance
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

	porcupines := make([][]float64, population)
	values := make([]float64, population)
	for i := range porcupines {
		porcupine := make([]float64, dimensions)
		for d := range porcupine {
			porcupine[d] = crestedPorcupineRangeLow + rand.Float64()*(crestedPorcupineRangeHigh-crestedPorcupineRangeLow)
		}
		porcupines[i] = porcupine
		values[i] = fn.Evaluate(porcupine)
	}

	best := append([]float64(nil), porcupines[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), porcupines[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range porcupines {
			r := rand.Float64()
			switch {
			case r < 0.25:
				for d := range porcupines[i] {
					porcupines[i][d] = porcupines[i][d] + rand.Float64()*(best[d]-porcupines[i][d])
				}
			case r < 0.5:
				rp := porcupines[randomIndexExcept(i)]
				for d := range porcupines[i] {
					porcupines[i][d] = porcupines[i][d] + rand.Float64()*(porcupines[i][d]-rp[d])
				}
			case r < 0.75:
				for d := range porcupines[i] {
					porcupines[i][d] = best[d] - rand.Float64()*(best[d]-porcupines[i][d])*(1-float64(steps)/float64(maxSteps))
				}
			default:
				for d := range porcupines[i] {
					porcupines[i][d] = best[d] + (rand.Float64()*2-1)*math.Abs(best[d]-porcupines[i][d])*0.5
				}
			}

			for d := range porcupines[i] {
				porcupines[i][d] = clamp(porcupines[i][d], crestedPorcupineRangeLow, crestedPorcupineRangeHigh)
			}
			values[i] = fn.Evaluate(porcupines[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), porcupines[i]...)
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
		Method:     "crested_porcupine",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
