package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"strconv"
	"strings"
)

const (
	saTsMaxSteps          = 2000
	saTsTemperature       = 10.0
	saTsCoolingRate       = 0.995
	saTsTabuSize          = 10
	saTsStepSize          = 0.5
	saTsTolerance         = 1e-6
	saTsRangeLow          = -5.0
	saTsRangeHigh         = 5.0
	saTsMaxRetries        = 10
	saTsCandidatesPerStep = 10
)

// SaTs minimizuje konfigurisanu benchmark funkciju koristeći Simulated Annealing sa Tabu memorijom: kandidat koji
// pada na nedavno posećenu (zaokruženu) poziciju iz tabu liste se odbacuje i generiše se nova perturbacija, a
// prihvaćeni kandidat prolazi kroz Metropolis kriterijum sa temperaturom koja opada tokom izvršavanja
// problem.Point je početna tačka pretrage
func SaTs(problem models.Problem) (result models.Result, err error) {

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

	maxSteps := saTsMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	temperature := saTsTemperature
	if v, ok := problem.Payload["temperature"].(float64); ok {
		temperature = v
	}

	coolingRate := saTsCoolingRate
	if v, ok := problem.Payload["cooling_rate"].(float64); ok {
		coolingRate = v
	}

	tabuSize := saTsTabuSize
	if v, ok := problem.Payload["tabu_size"].(float64); ok {
		tabuSize = int(v)
	}

	stepSize := saTsStepSize
	if v, ok := problem.Payload["step_size"].(float64); ok {
		stepSize = v
	}

	tolerance := saTsTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	roundPoint := func(p []float64) string {
		parts := make([]string, len(p))
		for i, v := range p {
			parts[i] = strconv.FormatFloat(v, 'f', 2, 64)
		}
		return strings.Join(parts, ",")
	}

	tabuList := make([]string, 0, tabuSize)
	isTabu := func(key string) bool {
		return slices.Contains(tabuList, key)
	}
	pushTabu := func(key string) {
		tabuList = append(tabuList, key)
		if len(tabuList) > tabuSize {
			tabuList = tabuList[1:]
		}
	}

	current := make([]float64, dimensions)
	for d := range current {
		current[d] = saTsRangeLow + rand.Float64()*(saTsRangeHigh-saTsRangeLow)
	}
	currentValue := fn.Evaluate(current)

	best := append([]float64(nil), current...)
	bestValue := currentValue

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		// Jedan kandidat po koraku daje jednopoteznoj pretrazi mnogo manje "pokušaja" po koraku nego
		// populacioni solveri (100 koraka naspram 100*population), pa je verovatnoća da se pronađe
		// poboljšanje veće od tolerance u bilo kom 100-koračnom prozoru premala — noImprove logika
		// prekida izvršavanje na dugim platoima iako bi pretraga uz više vremena nastavila da napreduje.
		// Uzorkovanje nekoliko kandidata po koraku i zadržavanje najboljeg (uz tabu proveru za svaki)
		// povećava tu verovatnoću bez menjanja temperature, cooling_rate ili step_size
		var bestCandidate []float64
		bestCandidateValue := math.Inf(1)
		for range saTsCandidatesPerStep {
			var candidate []float64
			for range saTsMaxRetries {
				candidate = make([]float64, dimensions)
				for d := range candidate {
					candidate[d] = clamp(current[d]+(rand.Float64()*2-1)*stepSize, saTsRangeLow, saTsRangeHigh)
				}
				if !isTabu(roundPoint(candidate)) {
					break
				}
			}
			candidateValue := fn.Evaluate(candidate)
			if candidateValue < bestCandidateValue {
				bestCandidateValue = candidateValue
				bestCandidate = candidate
			}
		}

		candidate := bestCandidate
		candidateValue := bestCandidateValue
		delta := candidateValue - currentValue

		if delta < 0 || rand.Float64() < math.Exp(-delta/temperature) {
			current = candidate
			currentValue = candidateValue
		}

		pushTabu(roundPoint(current))

		temperature *= coolingRate

		if currentValue < bestValue {
			bestValue = currentValue
			best = append([]float64(nil), current...)
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
		Method:     "sa_ts",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
