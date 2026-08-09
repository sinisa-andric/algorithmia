package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	whiteSharkPopulation = 20
	whiteSharkMaxSteps   = 1000
	whiteSharkTolerance  = 1e-6
	whiteSharkRangeLow   = -5.0
	whiteSharkRangeHigh  = 5.0
)

// WhiteShark minimizuje konfigurisanu benchmark funkciju koristeći White Shark Optimizer: brzina svake ajkule
// kombinuje inerciju i privlačenje ka dva nasumična člana jata, a povremeno ajkula umesto toga direktno prati miris
// plena krećući se ka trenutno najboljem rešenju; korak se prihvata samo ako poboljša rezultat
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func WhiteShark(problem models.Problem) (result models.Result, err error) {

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

	population := whiteSharkPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := whiteSharkMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := whiteSharkTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	sharks := make([][]float64, population)
	velocity := make([][]float64, population)
	values := make([]float64, population)
	for i := range sharks {
		shark := make([]float64, dimensions)
		for d := range shark {
			shark[d] = whiteSharkRangeLow + rand.Float64()*(whiteSharkRangeHigh-whiteSharkRangeLow)
		}
		sharks[i] = shark
		velocity[i] = make([]float64, dimensions)
		values[i] = fn.Evaluate(shark)
	}

	best := append([]float64(nil), sharks[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), sharks[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		ff := rand.Float64()*(0.1-1) + 1
		mv := rand.Float64()

		for i := range sharks {
			r1 := sharks[rand.IntN(population)]
			r2 := sharks[rand.IntN(population)]

			novi := make([]float64, dimensions)
			for d := range novi {
				velocity[i][d] = mv*velocity[i][d] + ff*rand.Float64()*(r1[d]-sharks[i][d]) + ff*rand.Float64()*(r2[d]-sharks[i][d])
				novi[d] = sharks[i][d] + velocity[i][d]
			}

			if rand.Float64() < 0.5 {
				for d := range novi {
					novi[d] = best[d] + rand.Float64()*(best[d]-sharks[i][d])
				}
			}

			for d := range novi {
				novi[d] = clamp(novi[d], whiteSharkRangeLow, whiteSharkRangeHigh)
			}

			if fn.Evaluate(novi) < fn.Evaluate(sharks[i]) {
				sharks[i] = novi
			}

			values[i] = fn.Evaluate(sharks[i])
			if values[i] < bestValue {
				bestValue = values[i]
				best = append([]float64(nil), sharks[i]...)
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
		Method:     "white_shark",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
