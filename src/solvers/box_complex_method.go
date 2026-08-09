package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	boxComplexMethodMaxSteps          = 1000
	boxComplexMethodTolerance         = 1e-6
	boxComplexMethodReflectionFactor  = 1.3
	boxComplexMethodLower             = -5.0
	boxComplexMethodUpper             = 5.0
	boxComplexMethodMaxContractions   = 5
	boxComplexMethodMaxBoundsPullback = 20
)

// boxWithinBounds proverava da li su sve komponente tačke unutar [lower,upper]
func boxWithinBounds(point []float64, lower, upper float64) bool {
	for _, v := range point {
		if v < lower || v > upper {
			return false
		}
	}
	return true
}

// boxCentroidExcluding računa centroid svih tačaka osim one na indeksu exclude
func boxCentroidExcluding(points [][]float64, exclude int) []float64 {

	dimensions := len(points[0])
	centroid := make([]float64, dimensions)
	count := 0
	for i, p := range points {
		if i == exclude {
			continue
		}
		for d := range centroid {
			centroid[d] += p[d]
		}
		count++
	}
	for d := range centroid {
		centroid[d] /= float64(count)
	}

	return centroid
}

// BoxComplexMethod minimizuje konfigurisanu benchmark funkciju koristeći Box-ov Complex metod: populacija tačaka
// ("complex") u [-5,5]^n, u svakom koraku se najgora tačka reflektuje preko centroida ostalih; KLJUČNA
// karakteristika metode je da se svaka tačka koja završi van granica povlači ka centroidu OSTALIH (ne prosto
// kliuje) dok ne uđe u granice
// problem.Point je početna tačka pretrage (projektovana u [-5,5] ako je van opsega)
func BoxComplexMethod(problem models.Problem) (result models.Result, err error) {

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

	maxSteps := boxComplexMethodMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := boxComplexMethodTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	dimensions := len(problem.Point)

	population := 2*dimensions + 2
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	reflectionFactor := boxComplexMethodReflectionFactor
	if v, ok := problem.Payload["reflection_factor"].(float64); ok {
		reflectionFactor = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	lower, upper := boxComplexMethodLower, boxComplexMethodUpper

	points := make([][]float64, population)
	points[0] = make([]float64, dimensions)
	for d := range points[0] {
		points[0][d] = clamp(problem.Point[d], lower, upper)
	}
	for i := 1; i < population; i++ {
		p := make([]float64, dimensions)
		for d := range p {
			p[d] = lower + rand.Float64()*(upper-lower)
		}
		points[i] = p
	}

	// KLJUČNA karakteristika Box metode: svaka tačka van granica se povlači ka centroidu OSTALIH dok ne uđe u
	// granice (generacija iznad već garantuje granice, ali ova provera čuva vernost algoritmu za svaki slučaj)
	for i := range points {
		guard := 0
		for !boxWithinBounds(points[i], lower, upper) && guard < boxComplexMethodMaxBoundsPullback {
			centroidOthers := boxCentroidExcluding(points, i)
			for d := range points[i] {
				points[i][d] = (points[i][d] + centroidOthers[d]) / 2
			}
			guard++
		}
	}

	values := make([]float64, population)
	for i := range points {
		values[i] = fn.Evaluate(points[i])
	}

	best := append([]float64(nil), points[0]...)
	bestValue := values[0]
	for i := 1; i < population; i++ {
		if values[i] < bestValue {
			bestValue = values[i]
			best = append([]float64(nil), points[i]...)
		}
	}

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		worstIdx := 0
		for i := 1; i < population; i++ {
			if values[i] > values[worstIdx] {
				worstIdx = i
			}
		}

		centroid := boxCentroidExcluding(points, worstIdx)

		nova := make([]float64, dimensions)
		for d := range nova {
			nova[d] = centroid[d] + reflectionFactor*(centroid[d]-points[worstIdx][d])
		}
		for d := range nova {
			nova[d] = clamp(nova[d], lower, upper)
		}

		novaValue := fn.Evaluate(nova)
		contractions := 0
		for novaValue >= values[worstIdx] && contractions < boxComplexMethodMaxContractions {
			for d := range nova {
				nova[d] = (nova[d] + centroid[d]) / 2
			}
			novaValue = fn.Evaluate(nova)
			contractions++
		}

		points[worstIdx] = nova
		values[worstIdx] = novaValue

		for i := range points {
			if values[i] < bestValue {
				bestValue = values[i]
				best = append([]float64(nil), points[i]...)
			}
			if includeTrajectory {
				trajectory = recordTrajectory(trajectory, steps, best, fn.Evaluate(best), false)
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
		trajectory = recordTrajectory(trajectory, steps, best, fn.Evaluate(best), true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "box_complex_method",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
