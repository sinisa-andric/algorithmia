package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	chameleonPopulation = 20
	chameleonMaxSteps   = 1000
	chameleonTolerance  = 1e-6
	chameleonRangeLow   = -5.0
	chameleonRangeHigh  = 5.0
)

// Chameleon minimizuje konfigurisanu benchmark funkciju koristeći Chameleon Swarm Algorithm: prva polovina
// izvršavanja simulira rotaciju očiju (šira pretraga ka nasumičnom kameleonu),
// druga polovina simulira gađanje jezikom — sve brže kretanje ka najboljem rešenju
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func Chameleon(problem models.Problem) (result models.Result, err error) {

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

	population := chameleonPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := chameleonMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := chameleonTolerance
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

	chameleons := make([][]float64, population)
	values := make([]float64, population)
	for i := range chameleons {
		chameleon := make([]float64, dimensions)
		for d := range chameleon {
			chameleon[d] = chameleonRangeLow + rand.Float64()*(chameleonRangeHigh-chameleonRangeLow)
		}
		chameleons[i] = chameleon
		values[i] = fn.Evaluate(chameleon)
	}

	best := append([]float64(nil), chameleons[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), chameleons[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		if float64(steps) < float64(maxSteps)/2 {
			for i := range chameleons {
				r := chameleons[randomIndexExcept(i)]
				for d := range chameleons[i] {
					// Čisto međusobno privlačenje (bez ikakve veze sa best-om) je kontraktivna
					// dinamika — udaljenost r[d]-chameleons[i][d] se eksponencijalno smanjuje iz
					// koraka u korak, pa roj brzo kolabira (gubi diverzitet) u proizvoljnu tačku
					// pre nego što faza gađanja jezikom uopšte počne (na pola izvršavanja), i tu
					// ostaje trajno zaglavljen jer dalji pomeraji postaju zanemarljivi — noImprove
					// logika prekine izvršavanje pre nego što faza eksploatacije uopšte dođe na red.
					// Ravnopravna privlačnost ka best-u drži roj u toku sa najboljim pronađenim
					// rešenjem tokom cele istraživačke faze, umesto da kolabira u slepu tačku.
					chameleons[i][d] = chameleons[i][d] + rand.Float64()*(r[d]-chameleons[i][d]) + rand.Float64()*(best[d]-chameleons[i][d])
				}
				for d := range chameleons[i] {
					chameleons[i][d] = clamp(chameleons[i][d], chameleonRangeLow, chameleonRangeHigh)
				}
				values[i] = fn.Evaluate(chameleons[i])
			}
		} else {
			v := 2 * (1 - float64(steps)/float64(maxSteps))
			for i := range chameleons {
				for d := range chameleons[i] {
					chameleons[i][d] = chameleons[i][d] + v*rand.Float64()*(best[d]-chameleons[i][d])
				}
				for d := range chameleons[i] {
					chameleons[i][d] = clamp(chameleons[i][d], chameleonRangeLow, chameleonRangeHigh)
				}
				values[i] = fn.Evaluate(chameleons[i])
			}
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), chameleons[i]...)
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
		Method:     "chameleon",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
