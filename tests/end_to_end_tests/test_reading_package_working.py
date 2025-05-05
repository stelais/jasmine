from jasmine.caustics.caustic_creator_cls import Caustic

caustic_object = Caustic(separation_s=0.8, mass_ratio_q=0.3, number_of_data_points=100)
type_of_caustics = caustic_object.define_topology()
assert type_of_caustics == 'resonant'