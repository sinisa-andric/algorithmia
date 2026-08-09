package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	mountainGazellePopulation = 20
	mountainGazelleMaxSteps   = 1000
	mountainGazelleTolerance  = 1e-6
	mountainGazelleRangeLow   = -5.0
	mountainGazelleRangeHigh  = 5.0
)

// MountainGazelle minimizuje konfigurisanu benchmark funkciju koristeći Mountain Gazelle Optimization: svaka gazela
// se ili kreće ka najboljem rešenju uz nasumičan poremećajni vektor i razliku dve gazele,
// ili stupa u socijalnu interakciju kombinujući razlike više nasumičnih članova stada i najboljeg
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func MountainGazelle(problem models.Problem) (result models.Result, err error) {

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

	population := mountainGazellePopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := mountainGazelleMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := mountainGazelleTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	gazelles := make([][]float64, population)
	values := make([]float64, population)
	for i := range gazelles {
		gazelle := make([]float64, dimensions)
		for d := range gazelle {
			gazelle[d] = mountainGazelleRangeLow + rand.Float64()*(mountainGazelleRangeHigh-mountainGazelleRangeLow)
		}
		gazelles[i] = gazelle
		values[i] = fn.Evaluate(gazelle)
	}

	best := append([]float64(nil), gazelles[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), gazelles[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range gazelles {
			cf := rand.Float64()
			r1 := gazelles[rand.IntN(population)]
			r2 := gazelles[rand.IntN(population)]
			r3 := gazelles[rand.IntN(population)]

			if rand.Float64() < 0.5 {
				m := make([]float64, dimensions)
				for d := range m {
					m[d] = mountainGazelleRangeLow + rand.Float64()*(mountainGazelleRangeHigh-mountainGazelleRangeLow)
				}
				for d := range gazelles[i] {
					gazelles[i][d] = best[d] + rand.Float64()*(gazelles[i][d]/2-m[d]) + cf*(r1[d]-r2[d])
				}
			} else {
				for d := range gazelles[i] {
					gazelles[i][d] += rand.Float64()*(r1[d]-r3[d]) + rand.Float64()*(best[d]-r2[d])
				}
			}

			for d := range gazelles[i] {
				gazelles[i][d] = clamp(gazelles[i][d], mountainGazelleRangeLow, mountainGazelleRangeHigh)
			}

			values[i] = fn.Evaluate(gazelles[i])
			if values[i] < bestValue {
				bestValue = values[i]
				best = append([]float64(nil), gazelles[i]...)
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
		Method:     "mountain_gazelle",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
