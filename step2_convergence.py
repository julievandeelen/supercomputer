import numpy as np
import functools
import math

from ema_workbench import (Constraint, RealParameter, ScalarOutcome, CategoricalParameter, ema_logging, MultiprocessingEvaluator, SequentialEvaluator )

from ema_workbench.connectors.vensim import VensimModel
from ema_workbench.em_framework.parameters import Scenario
from ema_workbench.em_framework.optimization import (HyperVolume,
                                                     EpsilonProgress)

get_10_percentile = functools.partial(np.percentile, q=10)


def get_last_outcome(outcome,time):
    index = np.where(time == 2100) #model runs until 2100
    last_outcome = outcome[index][0]
    return last_outcome

def get_SD(outcome):
    sd=np.std(outcome)
    return sd

# def constraint_biomass(biomass, time):
#     index = np.where(time == 2010)  # model runs from 2010 onwards
#     initial_biomass = biomass[index][0]
#     lowest_allowable= initial_biomass*0.95
#     return lambda biomass:min(0, biomass-lowest_allowable)

def lookup_list(time_range):
    list = np.arange(0, time_horizon, 2).tolist()
    return list

class MyVensimModel(VensimModel):

    def run_experiment(self, experiment):
        # 'Look up harvesting quota'
        # f"Proposed harvesting quota {t}", 0, 10) for t in range(45)]
        lookup = []
        decisions = []
        for t in range(45):
            value = experiment.pop(f"Proposed harvesting quota {t}")
            decisions.append(value)
            lookup.append((2010+2*t, value))
        experiment['Look up harvesting quota'] = lookup
        results = super(MyVensimModel, self).run_experiment(experiment)
        nvars = len(decisions)
        # decisions = np.array(decisions)
        results['Inertia'] = np.sum(np.abs(np.diff(decisions)) > 0.1)/float(nvars-1) #0.1
        return results

if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO )

    wd = './Vensim models'
    vensimModel = MyVensimModel("simpleModel", wd=wd,
                               model_file='model_thesis_V53_influence.vpmx')


    # ref_scen = {'Annual consumption predator': 1.4793951149982765, 'Annual fish consumption per capita': 1.4905486550829681e-05, 'Average sinking time': 370.3013239994961, 'Average weight per adult MF': 8.163981231164809e-07, 'Average weight per juvinile MF': 7.498010770635817e-08, 'Average weight per juvinile predator': 0.08803941989161089, 'Average weight per predator': 0.3005661047038175, 'C content copepods': 0.5636337034647753, 'Carbon loss at living depth': 0.2537575508087975, 'Carbon loss underway': 0.02603217216061074, 'Catchability mauriculus': 0.2674345193621034, 'Catchability myctophidae': 0.09160454099950202, 'Consumption by MF in bodyweight': 7.456476033371044, 'Consumption by zooplankton in bodyweight': 3.0206578369722217, 'Conversion factor to ppm': 2.1737585651871574, 'Costs regular fish': 286598976614.8163, 'Costs status quo mauriculus': 257759500713.4569, 'Costs status quo myctophidae': 506274788546.0016, 'Delay sedimentation': 10808.977719939434, 'Delay weathering': 9681.460562759215, 'Depth euphotic zone': 102.46912385661638, 'Downwelling water': 1471423549679066.2, 'Efficiency factor fisheries': 7.18989194140771, 'Export efficiency': 1.192072791610716, 'Female fraction': 0.4909442788307429, 'Fishmeal to fish factor': 3.26900775083997, 'Fraction grazed C ending up in surface': 0.5232532656449236, 'Fraction of migrating MF constant': 0.5055289045059983, 'Fraction spawning mauriculus vs myctophidae': 0.6932957553933005, 'Grazing in surface by MF': 0.3885139046595848, 'Growth period mauriculus': 1.0878551351986874, 'Growth period myctophidae': 0.6087958651569034, 'Growth period predator': 3.4642047908244384, 'Harvest information delay': 2.9578121442115552, 'Information delay risk perception': 5.695382164703988, 'Initial juvinile predator weight': 5.839995607598064, 'Initial phytoplankton': 1.0397842836972404, 'Initial predator weight': 4.253788258750164, 'Initial sediment C': 3929.418706172269, 'Initial surface C': 590.9158580374548, 'Initial weight adult MF': 10.097603609462515, 'Initial weight juvinile MF': 0.6115201539477855, 'Initial zooplankton': 4.555574856814983, 'Life expectancy mauriculus': 1.9918077295480447, 'Life expectancy myctophidae adult': 3.830594812155453, 'Living depth mauriculus': 250, 'Living depth myctophidae': 850, 'Other carbon fluxes': 4.825958307899499, 'Other food sources': 1.7339318018243035, 'Percentage discarded fish': 0.0939190455534377, 'Predator life expectancy': 5.36546666213312, 'Residence time deep carbon': 920.2004893248087, 'SWITCH lanternfish to mauriculus': 0.5231711302921653, 'Sale price regular fish': 2806596375722.9365, 'Share of aquaculture': 0.5194868458222378, 'Share of irreplaceable fishmeal': 0.8927608784038987, 'Spawning fraction': 0.21078810717128216, 'Specialist capacity building time': 4.815340319575954, 'Surface ocean': 386370786396128.94, 'Survived larvea': 160.28631326362049, 'Switch influence CO2 on phytoplankton': 2, 'Switch influence sunlight on phytoplankton': 2, 'Switch population growth': 3, 'Switch price change': 1, 'Switch profitability change MF fisheries': 1, 'Switch risk perception biomass': 1, 'Switch risk perception climate': 2, 'Switch risk reward mechanism': 1, 'Total atmospheric volume': 3.980817286603681e+18, 'Transfer velocity for GtC per year': 1.1811149601521738, 'Turnover time phytoplankton': 0.07823420422770139, 'Upwelling delay surface': 7.247654379638932, 'ppm conversion for ocean': 1.9631113980999195}
    ref_scen = {'C content copepods': 0.4953719065838029, 'Catchability myctophidae': 0.09394581802293928, 'Consumption by MF in bodyweight': 7.624438318949381, 'Consumption by zooplankton in bodyweight': 2.718235730961183, 'Conversion factor to ppm': 2.229903629345519, 'Depth euphotic zone': 108.6213435262702, 'Downwelling water': 1409652740145935.8, 'Female fraction': 0.4370404121552624, 'Fishmeal to fish factor': 4.706407865844137, 'Fraction grazed C ending up in surface': 0.5184917115127994, 'Fraction spawning mauriculus vs myctophidae': 0.6899077817258453, 'Grazing in surface by MF': 0.31492030081132383, 'Initial surface C': 604.541199511406, 'Initial weight adult MF': 9.748824700877329, 'Initial weight juvinile MF': 0.6715159261578895, 'Initial zooplankton': 3.5993934535877603, 'Residence time deep carbon': 904.1107205930556, 'SWITCH lanternfish to mauriculus': 0.5775324837154046, 'Spawning fraction': 0.19557565050837236, 'Surface ocean': 382517944113235.5, 'Switch price change': 1, 'Switch profitability change MF fisheries': 2, 'Switch risk reward mechanism': 3, 'Transfer velocity for GtC per year': 1.3001263281133304, 'Upwelling delay surface': 7.704994807574677}

    ref_scen = Scenario('ref_scen', **ref_scen)

    vensimModel.uncertainties = [#structural uncertainties
                                 CategoricalParameter('Switch risk reward mechanism', (1,2,3) ),
                                 CategoricalParameter('Switch profitability change MF fisheries', (1,2) ),#1
                                 CategoricalParameter('Switch price change', (1,2) ),
                                 # CategoricalParameter('Switch risk perception climate', (1,2,3) ),
                                 # CategoricalParameter('Switch risk perception biomass', (1,2) ),
                                 # CategoricalParameter('Switch influence sunlight on phytoplankton', (1,2) ),
                                 # CategoricalParameter('Switch influence CO2 on phytoplankton', (1,2) ),
                                 # CategoricalParameter('Switch population growth', (1,2,3) ),
                                 # CategoricalParameter("Living depth myctophidae", (550, 650, 750, 850, 950) ),  # 950
                                 # CategoricalParameter("Living depth mauriculus", (250, 350, 450) ),  # 350

                                 #parametric uncertainties
                                 RealParameter("SWITCH lanternfish to mauriculus", 0.4 , 1),  #1 **** #3
                                 RealParameter("Initial weight juvinile MF", 1*0.6 , 1*1.4),  #1 ****
                                 RealParameter("Initial weight adult MF", 9*0.8 , 9*1.2),  #9
                                 # RealParameter("Initial juvinile predator weight", 6*0.8, 6*1.2),  #6
                                 # RealParameter("Initial predator weight", 4*0.8, 4*1.2),  #4 ****
                                 # RealParameter("Average weight per predator", 0.3*0.8, 0.3*1.2),  #0.3
                                 # RealParameter("Average weight per juvinile predator", 0.08*0.8, 0.08*1.2),  #0.08
                                 RealParameter("Initial zooplankton", 4*0.8, 4*1.2),  #4 ** ****
                                 # RealParameter("Initial phytoplankton", 1*0.8, 1*1.2),  #1 ****
                                 RealParameter("Initial surface C", 600*0.95,600*1.05),  #600 #CHANGED FROM 20% to 10% ** ***
                                 # RealParameter("Initial sediment C", 3390*0.8, 3390*1.2),  #3390 ****
                                 RealParameter("Conversion factor to ppm", 2.0619 * 0.9, 2.0619 * 1.1),  # 2.0619 ****
                                 RealParameter("Surface ocean", 363000000000000*0.9, 363000000000000*1.1),  # 3.63*10^14 #CHANGED FROM 20% to 10% ** *** ****
                                 # RealParameter("Total atmospheric volume", 3990000000000000000*0.8, 3990000000000000000*1.2),  # 3.99*10^18
                                 RealParameter("Fraction grazed C ending up in surface", 0.4*0.6, 0.4*1.4),  # 0.4 **
                                 # RealParameter("Other carbon fluxes", 5*0.8, 5*1.2),  # 5
                                 # RealParameter("Carbon loss underway", 0.04*0.6, 0.04*1.4),  # 0.04
                                 # RealParameter("Carbon loss at living depth", 0.4*0.6, 0.4*1.4),  # 0.4
                                 # RealParameter("Average sinking time", 380*0.8, 380*1.2),  # 380
                                 # RealParameter("Delay sedimentation", 10000*0.8, 10000*1.2),  # 10000 ****
                                 # RealParameter("Delay weathering", 10000*0.8, 10000*1.2),  # 10000
                                 # RealParameter("Efficiency factor fisheries", 7 * 0.8, 7 * 1.2), #7
                                 # RealParameter("Growth period predator", 3*0.8, 3*1.2),  # 3
                                 # RealParameter("Other food sources", 2*0.8, 2*1.2),  # 2
                                 # RealParameter("Turnover time phytoplankton", 0.077*0.8, 0.077*1.2),  # 0.077 ****
                                 RealParameter("Consumption by zooplankton in bodyweight", 3*0.8, 3*1.2),  # 3
                                 # RealParameter("Harvest information delay", 3*0.8, 3*1.2),  # 3
                                 # RealParameter("Costs regular fish", 339000000000*0.8, 339000000000*1.2),  # 3.39e+11
                                 # RealParameter("Sale price regular fish", 3100000000000*0.8, 3100000000000*1.2),  # 3.1e+12
                                 # RealParameter("Costs status quo myctophidae", 450000000000*0.6, 450000000000*1.4),  # 450*10^9 ****
                                 # RealParameter("Costs status quo mauriculus", 300000000000*0.8, 300000000000*1.2),  # 225*10^9
                                 # RealParameter("Information delay risk perception", 5*0.8, 5*1.2),  # 5 ****
                                 # RealParameter("ppm conversion for ocean", 2.1*0.9, 2.1*1.1),  # 2.1 #CHANGED FROM 20% to 10% *** ****
                                 RealParameter("Downwelling water", 1700000000000000*0.8, 1700000000000000*1.2),  # 1.7*10^15 ***
                                 RealParameter("Residence time deep carbon", 1000*0.9, 1000*1.1),  # 1000 #CHANGED FROM 20% to 10% ***
                                 RealParameter("Upwelling delay surface", 8*0.9, 8*1.1),  # 8 #CHANGED FROM 20% to 10% ***
                                 # RealParameter("Share of aquaculture", 0.5*0.8, 0.5*1.2),  # 0.5 ****
                                 # RealParameter("Share of irreplaceable fishmeal", 0.8, 1),  # 0.1
                                 # RealParameter("Annual fish consumption per capita", 0.000017*0.8, 0.000017*1.2),  # 1.7*10^-5
                                 # RealParameter("Percentage discarded fish", 0.08*0.8, 0.08*1.2), #0.08
                                 # RealParameter("Specialist capacity building time", 5*0.8, 5*1.2),  # 5
                                 RealParameter("Catchability myctophidae", 0.14*0.6, 0.14*1.4),  # 0.14
                                 # RealParameter("Catchability mauriculus", 0.28*0.8, 0.28*1.2),  # 0.28
                                 # RealParameter("Fraction of migrating MF constant", 0.37*0.6, 0.37*1.4),  # 0.37 **
                                 RealParameter("Grazing in surface by MF", 0.4*0.6, 0.4*1.4),  # 0.4 ** ****
                                 RealParameter("C content copepods", 0.51*0.8, 0.51*1.2),  # 0.51
                                 RealParameter("Depth euphotic zone", 100*0.9, 100*1.1),  # 100#CHANGED FROM 40% to 25% *** ****
                                 # RealParameter("Total atmospheric volume", 3990000000000000000*0.8, 3990000000000000000*1.2),  # 3.99*10^18
                                 # RealParameter("Export efficiency", 0.97*0.6, 0.97*1.4),  # 0.97
                                 RealParameter("Transfer velocity for GtC per year", 1.12169*0.8, 1.12169*1.2),  # 1.12169 ***
                                 # RealParameter("Average weight per adult MF", 0.000001*0.8, 0.000001*1.2),  # 1*10^-6 ****
                                 # RealParameter("Average weight per juvinile MF", 0.00000007*0.8, 0.00000007*1.2),  # 0.07*10^-6
                                 # RealParameter("Life expectancy myctophidae adult", 4*0.8, 4*1.2),  # 4 ****
                                 # RealParameter("Life expectancy mauriculus", 2*0.8, 2*1.2),  # 2
                                 # RealParameter("Growth period myctophidae", 0.583*0.9, 0.583*1.1),  # 0.583
                                 # RealParameter("Growth period mauriculus", 1*0.9, 1*1.1),  # 1
                                 RealParameter("Consumption by MF in bodyweight", 7*0.8, 7*1.2),  # 7 **
                                 # RealParameter("Annual consumption predator", 1.6*0.8, 1.6*1.2),  # 1.6
                                 # RealParameter("Predator life expectancy", 6*0.8, 6*1.2),  # 6
                                 # RealParameter("Survived larvea", 147*0.8, 147*1.2),  # 147
                                 RealParameter("Spawning fraction", 0.18*0.8, 0.18*1.2),  # 0.18 ****
                                 RealParameter("Female fraction", 0.515*0.8, 0.515*1.2),  # 0.515
                                 RealParameter("Fraction spawning mauriculus vs myctophidae", 0.75*0.9, 0.75*1.1),  # 0.75
                                 RealParameter("Fishmeal to fish factor", 4*0.8, 4*1.2)  # 4
                                 ]

    vensimModel.levers = [RealParameter(f"Proposed harvesting quota {t}", 0, 6) for t in range(45)] #

    vensimModel.outcomes = [ScalarOutcome('Average food provision by MF', variable_name='Food provision by MF',
                                          kind=ScalarOutcome.MAXIMIZE, function=np.mean),
                            ScalarOutcome('Average vertical migration', variable_name='Total vertical migration', kind=ScalarOutcome.MAXIMIZE
                                   , function=np.mean),
                            ScalarOutcome('Biomass MF 10th percentile', variable_name='Biomass mesopelagic fish', kind=ScalarOutcome.MAXIMIZE
                                   , function=get_10_percentile),
                            ScalarOutcome('Final atmospheric C level', variable_name=['Atmospheric C', 'TIME'], kind=ScalarOutcome.MINIMIZE
                                          , function=get_last_outcome),
                            ScalarOutcome('Inertia', kind=ScalarOutcome.MINIMIZE)]


    convergence_metrics =[EpsilonProgress()]

    with MultiprocessingEvaluator(vensimModel) as evaluator:
        results, convergence = evaluator.optimize(nfe=800000, searchover='levers', convergence=convergence_metrics, #10
                                      epsilons=[0.05,] * len(vensimModel.outcomes) , Scenario=ref_scen) #0.001 #0.1


    results.to_excel('./Data/results_convergence7_supercomp3.xlsx')
    convergence.to_excel('./Data/conv_convergence7_supercomp3.xlsx')




