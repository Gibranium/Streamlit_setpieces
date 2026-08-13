"""Player evaluation dashboards.

Config (pair lists, INVERT maps, panel layouts) is lifted verbatim from the
EVALUATION-*.ipynb notebooks; the renderer is a restyled port of their
`percentile_of` / `cluster_positions` / `draw_paired_bars` / `plot_dashboard`.

Restyle vs the notebooks: volume bars use the app accent instead of #1565C0,
and quality bars use the app's truncated afmhot_r ramp so low percentiles stay
visible against the concrete background (the untruncated ramp goes near-white
at the bottom, which disappears on #D7D1CF).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


DASHBOARDS = {'AMW': {'label': 'Attacking midfielders & wingers',
         'pairs': {'finalization_pairs': [('Shots\nVolume', 'shots_30', None, True, 'shot'),
                                          ('npxG', 'npxG_30', None, False, 'shot'),
                                          ('xG per Shot', 'xg_per_shot', None, False, 'shot'),
                                          ('Shots\nVAEP', 'shots_vaep_30', None, False, 'shot'),
                                          ('Touches in Box\nVolume', 'inbox_30', None, True, 'z')],
                   'defensive_pairs': [('Defensive Actions\n(Value & Volume)',
                                        'def_actions_vaep_30',
                                        'defensive_actions_30',
                                        False,
                                        'def'),
                                       ("Opponents' Value\nResponsability",
                                        'defensive_vaep_padj',
                                        None,
                                        False,
                                        'val')],
                   'passing_pairs': [('Passes\n(Value & Volume)',
                                      'passes_vaep_30',
                                      'passes_30',
                                      False,
                                      'pass'),
                                     ('Progressive Passes\n(Value & Volume)',
                                      'passes_vaep_prog_30',
                                      'passes_prog_30',
                                      False,
                                      'prog'),
                                     ('Passes into Half Spaces or Bin14\n(Value & Volume)',
                                      'pass_into_zones_vaep_30',
                                      'pass_into_zones_30',
                                      False,
                                      'piz'),
                                     ('Turnovers\nVolume', 'turnovers_per_touch', None, True, 'to'),
                                     ('npxG Assisted', 'npxA_30', None, False, 'xa'),
                                     ('Passes & Carries\ninto Box (Volume)',
                                      'intobox_30',
                                      None,
                                      True,
                                      'box')],
                   'carrying_pairs': [('Carries\n(Value & Volume)',
                                       'carries_vaep_30',
                                       'carries_30',
                                       False,
                                       'carry'),
                                      ('Take-ons\nVolume', 'takeons_30', None, True, 'to'),
                                      ('Take-ons\nVAEP', 'takeons_vaep_30', None, False, 'to'),
                                      ('Standstill\nTake-on %', 'takeons_standstill_pct', None, False, 'to'),
                                      ('Receptions in Half Spaces or Bin14\n'
                                       '(Value of Reception + Following Action & Volume)',
                                       'receptions_zones_vaep_30',
                                       'receptions_zones_30',
                                       False,
                                       'rz'),
                                      ('After Receival\n(Value & Volume)',
                                       'after_receival_fwd_vaep_30',
                                       'after_receival_fwd_30',
                                       False,
                                       'ar'),
                                      ('Pass VAEP\nafter Carry',
                                       'pass_after_carry_vaep_30',
                                       None,
                                       False,
                                       'ac'),
                                      ('xG per Shot\nafter Carry',
                                       'xg_per_shot_after_carry',
                                       None,
                                       False,
                                       'ac')]},
         'invert': {'defensive_vaep_padj': True, 'turnovers_per_touch': True},
         'grid': {'nrows': 4, 'ncols': 12, 'hspace': 0.55, 'wspace': 0.45},
         'panels': [{'row': 1,
                     'cols': [0, 9],
                     'pairs': 'carrying_pairs',
                     'title': 'Carrying & Receiving',
                     'ylabel': True,
                     'legend': False,
                     'W': 0.6,
                     'intra_by_group': None},
                    {'row': 1,
                     'cols': [9, 12],
                     'pairs': 'defensive_pairs',
                     'title': 'Defending',
                     'ylabel': False,
                     'legend': True,
                     'W': 0.6,
                     'intra_by_group': None},
                    {'row': 2,
                     'cols': [0, 4],
                     'pairs': 'finalization_pairs',
                     'title': 'Finalization',
                     'ylabel': True,
                     'legend': False,
                     'W': 0.6,
                     'intra_by_group': None},
                    {'row': 2,
                     'cols': [4, 12],
                     'pairs': 'passing_pairs',
                     'title': 'Distribution',
                     'ylabel': False,
                     'legend': False,
                     'W': 0.6,
                     'intra_by_group': None}]},
 'CB': {'label': 'Centre-backs',
        'pairs': {'finalization_pairs': [('Shots\nVolume', 'shots_30', None, True, 'shot'),
                                         ('xG per Shot', 'xg_per_shot', None, False, 'shot')],
                  'carrying_pairs': [('Take-ons & Carries\n(Value & Volume)',
                                      'takeon_dribble_vaep_30',
                                      'takeon_dribble_30',
                                      False,
                                      'carry'),
                                     ('Carries into\nMiddle 3rd', 'carries_endmid_30', None, True, 'cbox'),
                                     ('Carries into\nFinal 3rd', 'carries_endfin_30', None, True, 'cbox'),
                                     ('After Receival Fwd\n(Value & Volume)',
                                      'after_receival_fwd_vaep_30',
                                      'after_receival_fwd_30',
                                      False,
                                      'ar')],
                  'defensive_pairs': [('Defensive Actions\nAvg. Height',
                                       'def_actions_avg_x',
                                       None,
                                       False,
                                       'da'),
                                      ('Defensive Actions\n(Value & Volume)',
                                       'def_actions_vaep_30',
                                       'defensive_actions_30',
                                       False,
                                       'da'),
                                      ('Tackles + Interceptions\nVolume',
                                       'tackles_interceptions_30',
                                       None,
                                       True,
                                       'da'),
                                      ("Opponents' Value\nResponsibility",
                                       'defensive_vaep_padj',
                                       None,
                                       False,
                                       'opp'),
                                      ('Tackle True\nWin Rate', 'tackles_true_win_rate', None, False, 'twr'),
                                      ('Fouls\nVolume', 'fouls_30', None, True, 'foul'),
                                      ('Tactical\nFoul %', 'tactical_fouls_pct', None, False, 'foul'),
                                      ('Turnovers\nFaced', 'turnovers_faced_30', None, True, 'tdef'),
                                      ('Against Turnover Def Actions\n(Value & Volume)',
                                       'turnover_def_actions_vaep_30',
                                       'turnover_def_actions_30',
                                       False,
                                       'tdef')],
                  'passing_pairs': [('Passes Def 3rd\n(Value & Volume)',
                                     'passes_vaep_def_30',
                                     'passes_def_30',
                                     False,
                                     'pdef'),
                                    ('Prog Passes Def 3rd\n(Value & Volume)',
                                     'passes_vaep_prog_def_30',
                                     'passes_prog_def_30',
                                     False,
                                     'pdef'),
                                    ('Def 3rd\nPAx', 'PAx100_def', None, False, 'pdef'),
                                    ('Passes Mid 3rd\n(Value & Volume)',
                                     'passes_vaep_mid_30',
                                     'passes_mid_30',
                                     False,
                                     'pmid'),
                                    ('Prog Passes Mid 3rd\n(Value & Volume)',
                                     'passes_vaep_prog_mid_30',
                                     'passes_prog_mid_30',
                                     False,
                                     'pmid'),
                                    ('Mid 3rd\nPAx', 'PAx100_mid', None, False, 'pmid'),
                                    ('Passes into\nFinal 3rd Volume', 'passes_endfin_30', None, True, 'pfin'),
                                    ('Final 3rd\nPAx', 'PAx100_endfin', None, False, 'pfin'),
                                    ('Under Pressure\nPAx Change',
                                     'under_pressure_change_PAx',
                                     None,
                                     False,
                                     'pr'),
                                    ('Turnovers\nVolume', 'turnovers_30', None, True, 'to')]},
        'invert': {'defensive_vaep_padj': True, 'turnovers_30': True, 'fouls_30': True},
        'grid': {'nrows': 4, 'ncols': 12, 'hspace': 0.55, 'wspace': 0.55},
        'panels': [{'row': 1,
                    'cols': [0, 2],
                    'pairs': 'finalization_pairs',
                    'title': 'Finalization',
                    'ylabel': True,
                    'legend': False,
                    'W': 0.6,
                    'intra_by_group': None},
                   {'row': 1,
                    'cols': [2, 12],
                    'pairs': 'passing_pairs',
                    'title': 'Distribution',
                    'ylabel': False,
                    'legend': False,
                    'W': 0.6,
                    'intra_by_group': {'pdef': 1.32, 'pmid': 1.32}},
                   {'row': 2,
                    'cols': [0, 4],
                    'pairs': 'carrying_pairs',
                    'title': 'Carrying & Receiving',
                    'ylabel': True,
                    'legend': False,
                    'W': 0.6,
                    'intra_by_group': None},
                   {'row': 2,
                    'cols': [4, 12],
                    'pairs': 'defensive_pairs',
                    'title': 'Defending',
                    'ylabel': False,
                    'legend': False,
                    'W': 0.6,
                    'intra_by_group': {'da': 1.0, 'tdef': 1.0}}]},
 'CDM': {'label': 'Central midfielders',
         'pairs': {'finalization_pairs': [('Shots\nVolume', 'shots_30', None, True, 'shot'),
                                          ('xG per Shot', 'xg_per_shot', None, False, 'shot'),
                                          ('npxG\nAssisted', 'npxA_30', None, False, 'fin_q')],
                   'carrying_pairs': [('Take-ons & Carries\n(Value & Volume)',
                                       'takeon_dribble_vaep_30',
                                       'takeon_dribble_30',
                                       False,
                                       'carry'),
                                      ('Carries into\nBox', 'carries_box_30', None, True, 'cbox'),
                                      ('Carries into\nFinal 3rd', 'carries_endfin_30', None, True, 'cbox'),
                                      ('Receptions in HS / Bin14\n(Value & Volume)',
                                       'receptions_zones_vaep_30',
                                       'receptions_zones_30',
                                       False,
                                       'rz'),
                                      ('After Receival Fwd\n(Value & Volume)',
                                       'after_receival_fwd_vaep_30',
                                       'after_receival_fwd_30',
                                       False,
                                       'ar')],
                   'defensive_pairs': [('Defensive Actions\n(Value & Volume)',
                                        'def_actions_vaep_30',
                                        'defensive_actions_30',
                                        False,
                                        'da'),
                                       ('Tackles + Interceptions\nVolume',
                                        'tackles_interceptions_30',
                                        None,
                                        True,
                                        'da'),
                                       ("Opponents' Value\nResponsibility",
                                        'defensive_vaep_padj',
                                        None,
                                        False,
                                        'opp'),
                                       ('Tackle True\nWin Rate', 'tackles_true_win_rate', None, False, 'twr'),
                                       ('Fouls\nVolume', 'fouls_30', None, True, 'foul'),
                                       ('Tactical\nFoul %', 'tactical_fouls_pct', None, False, 'foul'),
                                       ('Turnovers\nFaced', 'turnovers_faced_30', None, True, 'tdef'),
                                       ('Against Turnover Def Actions\n(Value & Volume)',
                                        'turnover_def_actions_vaep_30',
                                        'turnover_def_actions_30',
                                        False,
                                        'tdef')],
                   'passing_pairs': [('Passes Def 3rd\n(Value & Volume)',
                                      'passes_vaep_def_30',
                                      'passes_def_30',
                                      False,
                                      'pdef'),
                                     ('Prog Passes Def 3rd\n(Value & Volume)',
                                      'passes_vaep_prog_def_30',
                                      'passes_prog_def_30',
                                      False,
                                      'pdef'),
                                     ('Def 3rd\nPAx', 'PAx100_def', None, False, 'pdef'),
                                     ('Passes Mid 3rd\n(Value & Volume)',
                                      'passes_vaep_mid_30',
                                      'passes_mid_30',
                                      False,
                                      'pmid'),
                                     ('Prog Passes Mid 3rd\n(Value & Volume)',
                                      'passes_vaep_prog_mid_30',
                                      'passes_prog_mid_30',
                                      False,
                                      'pmid'),
                                     ('Mid 3rd\nPAx', 'PAx100_mid', None, False, 'pmid'),
                                     ('Passes into\nFinal 3rd Volume',
                                      'passes_endfin_30',
                                      None,
                                      True,
                                      'pfin'),
                                     ('Final 3rd\nPAx', 'PAx100_endfin', None, False, 'pfin'),
                                     ('Passes into\nBox Volume', 'passes_box_30', None, True, 'pbox'),
                                     ('Into Box\nPAx', 'PAx100_box', None, False, 'pbox'),
                                     ('Under Pressure\nPAx Change',
                                      'under_pressure_change_PAx',
                                      None,
                                      False,
                                      'pr'),
                                     ('Turnovers\nVolume', 'turnovers_30', None, True, 'to')]},
         'invert': {'defensive_vaep_padj': True, 'turnovers_30': True, 'fouls_30': True},
         'grid': {'nrows': 4, 'ncols': 12, 'hspace': 0.55, 'wspace': 0.55},
         'panels': [{'row': 1,
                     'cols': [0, 3],
                     'pairs': 'finalization_pairs',
                     'title': 'Finalization',
                     'ylabel': True,
                     'legend': False,
                     'W': 0.6,
                     'intra_by_group': None},
                    {'row': 1,
                     'cols': [3, 12],
                     'pairs': 'passing_pairs',
                     'title': 'Distribution',
                     'ylabel': False,
                     'legend': True,
                     'W': 0.6,
                     'intra_by_group': {'pdef': 1.32, 'pmid': 1.32}},
                    {'row': 2,
                     'cols': [0, 5],
                     'pairs': 'carrying_pairs',
                     'title': 'Carrying & Receiving',
                     'ylabel': True,
                     'legend': False,
                     'W': 0.6,
                     'intra_by_group': None},
                    {'row': 2,
                     'cols': [5, 12],
                     'pairs': 'defensive_pairs',
                     'title': 'Defending',
                     'ylabel': False,
                     'legend': False,
                     'W': 0.6,
                     'intra_by_group': {'da': 1.0, 'tdef': 1.0}}]},
 'GK': {'label': 'Goalkeepers',
        'pairs': {'all_pairs': [('Overall', 'goals_prevented_30', 'shots_faced_30'),
                                ('Shots 8s after a Transition',
                                 'goals_prevented_transition_30',
                                 'shots_faced_transition_30'),
                                ('Shots in a 8s Window\nSince Last Faced',
                                 'gp_rebound_lt8s_30',
                                 'shots_rebound_lt8s_30'),
                                ('Shots between 9s and 1m\nSince Last Faced',
                                 'gp_same_attack_8_60s_30',
                                 'shots_same_attack_8_60s_30'),
                                ('Shots between 1m and 6m\nSince Last Faced',
                                 'gp_follow_up_1m_6m_30',
                                 'shots_follow_up_1m_6m_30'),
                                ('Shots after more than 6m\nSince Last Faced',
                                 'gp_rested_6m_plus_30',
                                 'shots_rested_6m_plus_30'),
                                ('First Shot of the Period', 'gp_period_start', 'shots_period_start')],
                  'block_bottom_pairs': [('Touches (Volume)', 'touches', None),
                                         ('GK Actions (Value & Volume)', 'gk_vaep', 'gk_vaep_vol'),
                                         ('Claims & Punches\nper Cross Faced', 'claims_punches_rt', None),
                                         ('Def. Actions\nDistance from the Goal',
                                          'def_actions_distance',
                                          None)],
                  'block_right_pairs': [('Short Passes\n(Above Expectations and Volume)',
                                         'shortballs_PAx100',
                                         'shortballs'),
                                        ('Short Passes\ninto danger', 'short_passes_indanger_pct', None),
                                        ('Long Balls\n(Above Expectations and Volume)',
                                         'longballs_PAx100',
                                         'longballs'),
                                        ('Long Balls\ninto danger', 'long_passes_indanger_pct', None)]},
        'invert': {'short_passes_indanger_pct': True, 'long_passes_indanger_pct': True},
        'grid': {'nrows': 4, 'ncols': 4, 'hspace': 0.45, 'wspace': 0.25},
        'panels': [{'row': 1,
                    'cols': [0, 4],
                    'pairs': 'all_pairs',
                    'title': 'Goals Prevented and Volume of Shots Faced in Specific Situation',
                    'ylabel': True,
                    'legend': True,
                    'W': 0.5,
                    'intra_by_group': None},
                   {'row': 2,
                    'cols': [0, 2],
                    'pairs': 'block_bottom_pairs',
                    'title': 'Partecipation and Other Goalkeeping Metrics',
                    'ylabel': True,
                    'legend': False,
                    'W': 0.5,
                    'intra_by_group': None},
                   {'row': 2,
                    'cols': [2, 4],
                    'pairs': 'block_right_pairs',
                    'title': 'Distribution',
                    'ylabel': False,
                    'legend': False,
                    'W': 0.5,
                    'intra_by_group': None}]},
 'ST': {'label': 'Strikers',
        'pairs': {'finalization_pairs': [('Shots\nper Touch', 'shots_per_touch', None, False, 'a'),
                                         ('Footed Shots\nVolume', 'shots_foot_30', None, True, 'foot'),
                                         ('Footed\nxG', 'npxG_foot_30', None, False, 'foot'),
                                         ('Footed\nxG per Shot', 'xg_per_shot_foot', None, False, 'foot'),
                                         ('Footed Shots\nVAEP', 'shots_vaep_foot_30', None, False, 'foot'),
                                         ('Headed Shots\nVolume', 'shots_head_30', None, True, 'head'),
                                         ('Headed\nxG', 'npxG_head_30', None, False, 'head'),
                                         ('Headed\nxG per Shot', 'xg_per_shot_head', None, False, 'head'),
                                         ('Headed Shots\nVAEP', 'shots_vaep_head_30', None, False, 'head'),
                                         ('Touches in Box\nVolume', 'inbox_30', None, True, 'z')],
                  'defensive_pairs': [('Defensive Actions\n(Value & Volume)',
                                       'def_actions_vaep_30',
                                       'defensive_actions_30',
                                       False,
                                       'def'),
                                      ("Opponents' Value\nResponsability",
                                       'defensive_vaep_padj',
                                       None,
                                       False,
                                       'val')],
                  'passing_pairs': [('Passes\n(Value & Volume)',
                                     'passes_vaep_30',
                                     'passes_30',
                                     False,
                                     'pass'),
                                    ('Progressive Passes\n(Value & Volume)',
                                     'passes_vaep_prog_30',
                                     'passes_prog_30',
                                     False,
                                     'prog'),
                                    ('Turnovers\nVolume', 'turnovers_per_touch', None, True, 'to'),
                                    ('npxG Assisted', 'npxA_30', None, False, 'xa'),
                                    ('Passes & Carries\ninto Box (Volume)', 'intobox_30', None, True, 'box')],
                  'carrying_pairs': [('Carries\n(Value & Volume)',
                                      'carries_vaep_30',
                                      'carries_30',
                                      False,
                                      'carry'),
                                     ('Take-ons\nVolume', 'takeons_30', None, True, 'to'),
                                     ('Take-ons\nVAEP', 'takeons_vaep_30', None, False, 'to'),
                                     ('Standstill\nTake-on %', 'takeons_standstill_pct', None, False, 'to'),
                                     ('Long Balls Received\n(Volume & Meaningful Possession Kept %)',
                                      'long_ball_mp_pct',
                                      'long_balls_received_30',
                                      False,
                                      'lb'),
                                     ('Pass VAEP\nafter Carry',
                                      'pass_after_carry_vaep_30',
                                      None,
                                      False,
                                      'ac'),
                                     ('xG per Shot\nafter Carry',
                                      'xg_per_shot_after_carry',
                                      None,
                                      False,
                                      'ac')]},
        'invert': {'defensive_vaep_padj': True, 'turnovers_per_touch': True},
        'grid': {'nrows': 4, 'ncols': 12, 'hspace': 0.55, 'wspace': 0.6},
        'panels': [{'row': 1,
                    'cols': [0, 9],
                    'pairs': 'finalization_pairs',
                    'title': 'Finalization',
                    'ylabel': True,
                    'legend': False,
                    'W': 0.5,
                    'intra_by_group': None},
                   {'row': 1,
                    'cols': [9, 12],
                    'pairs': 'defensive_pairs',
                    'title': 'Defending',
                    'ylabel': False,
                    'legend': True,
                    'W': 0.5,
                    'intra_by_group': None},
                   {'row': 2,
                    'cols': [0, 5],
                    'pairs': 'passing_pairs',
                    'title': 'Distribution',
                    'ylabel': True,
                    'legend': False,
                    'W': 0.62,
                    'intra_by_group': None},
                   {'row': 2,
                    'cols': [5, 12],
                    'pairs': 'carrying_pairs',
                    'title': 'Carrying & Receiving',
                    'ylabel': False,
                    'legend': False,
                    'W': 0.62,
                    'intra_by_group': None}]},
 'WB': {'label': 'Full-backs & wing-backs',
        'pairs': {'finalization_pairs': [('Shots\nVolume', 'shots_30', None, True, 'shot'),
                                         ('xG per Shot', 'xg_per_shot', None, False, 'shot'),
                                         ('npxG\nAssisted', 'npxA_30', None, False, 'fin_q')],
                  'carrying_pairs': [('Take-ons & Carries\n(Value & Volume)',
                                      'takeon_dribble_vaep_30',
                                      'takeon_dribble_30',
                                      False,
                                      'carry'),
                                     ('Carries\ninto Box', 'carries_box_30', None, True, 'cbox'),
                                     ('Carries into\nFinal 3rd', 'carries_endfin_30', None, True, 'cbox'),
                                     ('Receptions in HS / Bin14\n(Value & Volume)',
                                      'receptions_zones_vaep_30',
                                      'receptions_zones_30',
                                      False,
                                      'rz'),
                                     ('After Receival Fwd\n(Value & Volume)',
                                      'after_receival_fwd_vaep_30',
                                      'after_receival_fwd_30',
                                      False,
                                      'ar')],
                  'defensive_pairs': [('Def. Actions\n(Value & Volume)',
                                       'def_actions_vaep_30',
                                       'defensive_actions_30',
                                       False,
                                       'da'),
                                      ('Tackles + Interceptions\nVolume',
                                       'tackles_interceptions_30',
                                       None,
                                       True,
                                       'da'),
                                      ("Opponents' Value\nResponsibility",
                                       'defensive_vaep_padj',
                                       None,
                                       False,
                                       'opp'),
                                      ('Tackle True\nWin Rate', 'tackles_true_win_rate', None, False, 'twr'),
                                      ('Fouls\nVolume', 'fouls_30', None, True, 'foul'),
                                      ('Tactical\nFoul %', 'tactical_fouls_pct', None, False, 'foul'),
                                      ('Turnovers\nFaced', 'turnovers_faced_30', None, True, 'tdef'),
                                      ('Against Turnover\nDef. Actions\n(Value & Volume)',
                                       'turnover_def_actions_vaep_30',
                                       'turnover_def_actions_30',
                                       False,
                                       'tdef')],
                  'passing_pairs': [('Passes Def 3rd\n(Value & Volume)',
                                     'passes_vaep_def_30',
                                     'passes_def_30',
                                     False,
                                     'pdef'),
                                    ('Prog Passes Def 3rd\n(Value & Volume)',
                                     'passes_vaep_prog_def_30',
                                     'passes_prog_def_30',
                                     False,
                                     'pdef'),
                                    ('Def 3rd\nPAx', 'PAx100_def', None, False, 'pdef'),
                                    ('Passes Mid 3rd\n(Value & Volume)',
                                     'passes_vaep_mid_30',
                                     'passes_mid_30',
                                     False,
                                     'pmid'),
                                    ('Prog Passes Mid 3rd\n(Value & Volume)',
                                     'passes_vaep_prog_mid_30',
                                     'passes_prog_mid_30',
                                     False,
                                     'pmid'),
                                    ('Mid 3rd\nPAx', 'PAx100_mid', None, False, 'pmid'),
                                    ('Passes into\nFinal 3rd Volume', 'passes_endfin_30', None, True, 'pfin'),
                                    ('Final 3rd\nPAx', 'PAx100_endfin', None, False, 'pfin'),
                                    ('Passes into\nBox Volume', 'passes_box_30', None, True, 'pbox'),
                                    ('Into Box\nPAx', 'PAx100_box', None, False, 'pbox'),
                                    ('Under Pressure\nPAx Change',
                                     'under_pressure_change_PAx',
                                     None,
                                     False,
                                     'pr'),
                                    ('Turnovers\nVolume', 'turnovers_30', None, True, 'to')],
                  'crosses_pairs': [('Crosses\nVolume', 'crosses_30', None, True, 'cross'),
                                    ('Crosses\nValue 5s', 'avg_vaep_difference', None, False, 'cross'),
                                    ('Crosses %\nProduct in 5s', 'shot_5 %', None, False, 'cross')]},
        'invert': {'defensive_vaep_padj': True, 'turnovers_30': True, 'fouls_30': True},
        'grid': {'nrows': 4, 'ncols': 12, 'hspace': 0.55, 'wspace': 0.55},
        'panels': [{'row': 1,
                    'cols': [0, 3],
                    'pairs': 'finalization_pairs',
                    'title': 'Finalization',
                    'ylabel': True,
                    'legend': False,
                    'W': 0.6,
                    'intra_by_group': None},
                   {'row': 1,
                    'cols': [3, 12],
                    'pairs': 'passing_pairs',
                    'title': 'Distribution',
                    'ylabel': False,
                    'legend': False,
                    'W': 0.6,
                    'intra_by_group': {'pdef': 1.32, 'pmid': 1.32}},
                   {'row': 2,
                    'cols': [0, 4],
                    'pairs': 'carrying_pairs',
                    'title': 'Carrying & Receiving',
                    'ylabel': True,
                    'legend': False,
                    'W': 0.6,
                    'intra_by_group': None},
                   {'row': 2,
                    'cols': [4, 6],
                    'pairs': 'crosses_pairs',
                    'title': 'Crossing',
                    'ylabel': False,
                    'legend': False,
                    'W': 0.6,
                    'intra_by_group': {'da': 1.0, 'tdef': 1.0}},
                   {'row': 2,
                    'cols': [6, 12],
                    'pairs': 'defensive_pairs',
                    'title': 'Defending',
                    'ylabel': False,
                    'legend': False,
                    'W': 0.6,
                    'intra_by_group': {'da': 1.0, 'tdef': 1.0}}]}}


# ──────────────────────────────────────────────
# Value helpers — parquet hands back numpy arrays for the list columns
# (competition_id, season_id, and player_name on the GK export), so these
# accept ndarray as well as list/tuple.
# ──────────────────────────────────────────────
SEQ = (list, tuple, np.ndarray, pd.Series)


def unwrap(v):
    return v[0] if isinstance(v, SEQ) and len(v) else v


def clean_str(v, default='Unknown'):
    v = unwrap(v)
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return default
    return str(v)


def as_list(v):
    """Sorted unique strings from a scalar or sequence cell, NaNs dropped."""
    if isinstance(v, SEQ):
        return sorted({str(x) for x in v if not (isinstance(x, float) and pd.isna(x))})
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return []
    return [str(v)]


def ordinal(n):
    n = int(round(n))
    suf = 'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f'{n}{suf}'


def fmt_minutes(mp):
    if pd.isna(mp):
        return 'n/a'
    total = int(round(mp * 60))
    return f'{total // 60}:{total % 60:02d}'


def stint_label(row):
    return (f"{clean_str(row['player_name'])} — {clean_str(row['team_name'])} "
            f"({'/'.join(as_list(row['season_id']))})")


# ──────────────────────────────────────────────
# Percentiles — same definition as the notebooks: share of the pool strictly
# below the player's value, flipped for metrics where lower is better.
# ──────────────────────────────────────────────
def percentile_of(frame, pool, row_id, metric, invert=False):
    if metric is None or metric not in pool.columns:
        return np.nan
    vals = pool[metric].dropna().astype(float)
    if not len(vals):
        return np.nan
    x = frame.loc[row_id, metric]
    if pd.isna(x):
        return np.nan
    pct = (vals < float(x)).mean() * 100
    return 100 - pct if invert else pct


def cluster_positions(pairs, intra=0.65, gap=1.6, intra_by_group=None):
    """Bars in the same group sit `intra` apart, different groups `gap` apart."""
    intra_by_group = intra_by_group or {}
    xs, pos, prev = [], 0.0, None
    for item in pairs:
        group = item[4] if len(item) > 4 else None
        if prev is not None:
            pos += intra_by_group.get(group, intra) if (group is not None and group == prev) else gap
        xs.append(pos)
        prev = group
    return np.array(xs)


def draw_paired_bars(ax, frame, pool, row_id, pairs, title, invert, style,
                     ylabel=False, show_legend=True, W=0.5, intra_by_group=None):
    # The goalkeepers notebook has its own renderer: evenly spaced np.arange
    # positions, w=0.38, no group clustering and no x-limit clamp. Its pairs are
    # 3-tuples (no hatch flag, no group), which is exactly what distinguishes it
    # from the five outfield notebooks' 5-tuples.
    gk_layout = len(pairs[0]) <= 3
    if gk_layout:
        x, W, pad, value_fontsize = np.arange(len(pairs), dtype=float), 0.38, 0.0, 12
    else:
        x = cluster_positions(pairs, intra_by_group=intra_by_group)
        pad, value_fontsize = 0.03, 11
    accent, ink, muted, track, edge = (style['ACCENT'], style['INK'], style['MUTED'],
                                       style['TRACK'], style['BG'])
    reg, semi = style['fe_regular'], style['fe_semibold']

    def bar(xi, pct, raw, color, hatch, decimals):
        ax.bar(xi, 100, W, color=track, edgecolor=style['GRID'], linewidth=0.6, zorder=1)
        if np.isnan(pct):
            return False
        ax.bar(xi, pct, W, color=color, edgecolor=edge, linewidth=0.8, hatch=hatch, zorder=2)
        ax.text(xi, pct + 1.5, f'{ordinal(pct)}\n{raw:.{decimals}f}', ha='center', va='bottom',
                fontsize=value_fontsize, fontname=reg, color=ink)
        return True

    any_data = False
    for i, item in enumerate(pairs):
        label, qual_m, vol_m = item[0], item[1], item[2]
        hatch_single = item[3] if len(item) > 3 else False
        single = vol_m is None

        q_off = 0.0 if single else -W / 2 - pad
        q_pct = percentile_of(frame, pool, row_id, qual_m, invert.get(qual_m, False))
        q_raw = frame.loc[row_id, qual_m] if qual_m in frame.columns else np.nan
        volume_only = single and hatch_single
        q_color = accent if (volume_only or np.isnan(q_pct)) else style['afm_at'](q_pct / 100)
        any_data |= bar(x[i] + q_off, q_pct, q_raw, q_color,
                        '///' if volume_only else None, 3)

        if not single:
            v_pct = percentile_of(frame, pool, row_id, vol_m, False)
            v_raw = frame.loc[row_id, vol_m] if vol_m in frame.columns else np.nan
            any_data |= bar(x[i] + W / 2 + pad, v_pct, v_raw, accent, '///', 2)

    if not gk_layout:
        ax.set_xlim(x[0] - (W / 2 + 0.5), x[-1] + (W / 2 + 0.5))
    ax.set_ylim(0, 116)
    ax.set_xticks(x)
    ax.set_xticklabels([p[0] for p in pairs], fontsize=8, fontname=reg, color=muted)
    ax.tick_params(axis='y', labelsize=8, colors=muted)
    if ylabel:
        ax.set_ylabel('Percentile', fontname=reg, color=muted, fontsize=10)
    ax.set_title(title, fontsize=12, fontname=semi, color=ink, loc='left')
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(style['GRID'])

    if any_data and show_legend:
        ax.legend(handles=[Patch(facecolor=style['afm_at'](0.7), edgecolor=edge, label='Quality'),
                           Patch(facecolor=accent, edgecolor=edge, hatch='///', label='Volume')],
                  fontsize=8, loc='upper right', bbox_to_anchor=(1, 1.15), framealpha=0.6)
    elif not any_data:
        ax.text(0.5, 0.5, 'No data for this stint', ha='center', va='center',
                transform=ax.transAxes, color=muted, fontsize=12, fontname=reg)


def plot_dashboard(frame, pool, row_id, group, style):
    """Recreate the notebook dashboard for one stint against a chosen pool."""
    cfg = DASHBOARDS[group]
    grid, row = cfg['grid'], frame.loc[row_id]
    reg, semi, display = style['fe_regular'], style['fe_semibold'], style['fe_display']

    fig = plt.figure(figsize=(24, 13), facecolor=style['BG'])
    gs = fig.add_gridspec(nrows=grid['nrows'], ncols=grid['ncols'],
                          height_ratios=[0.14, 1, 1, 0.06],
                          hspace=grid['hspace'], wspace=grid['wspace'])

    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    ax_title.text(0.5, 0.98, clean_str(row['player_name']), size=30, ha='center', va='center',
                  color=style['INK'], fontname=display)
    meta = ' — '.join(x for x in [
        clean_str(row['team_name']),
        '/'.join(as_list(row['season_id'])),
        clean_str(row['position'], group),
        fmt_minutes(row['minutes_played']),
        ', '.join(as_list(row['competition_id'])),
    ] if x)
    ax_title.text(0.5, 0.05, meta, size=18, ha='center', va='center',
                  color=style['MUTED'], fontname=reg)
    # dropped below the meta line: long competition lists reach the right edge
    ax_title.text(0.99, -0.38, f'pool: {len(pool)} stints', size=9, ha='right', va='center',
                  color=style['MUTED'], fontname=reg)

    for panel in cfg['panels']:
        ax = fig.add_subplot(gs[panel['row'], panel['cols'][0]:panel['cols'][1]])
        ax.set_facecolor(style['BG'])
        draw_paired_bars(ax, frame, pool, row_id, cfg['pairs'][panel['pairs']],
                         panel['title'], cfg['invert'], style,
                         ylabel=panel['ylabel'], show_legend=panel['legend'],
                         W=panel['W'], intra_by_group=panel['intra_by_group'])

    ax_end = fig.add_subplot(gs[grid['nrows'] - 1, :])
    ax_end.axis('off')
    ax_end.text(0.5, 0.5, 'X: @gualanodavide | Bluesky: @gualanodavide.bsky.social | '
                'Linkedin: www.linkedin.com/in/davide-gualano-a2454b187 | '
                'Newsletter: the-cutback.beehiiv.com',
                size=10, ha='center', va='center', color=style['MUTED'], fontname=reg)
    return fig


def pretty_metric(col):
    """'passes_vaep_prog_def_30__pctile' -> 'Passes Vaep Prog Def 30'."""
    return col.replace('__pctile', '').replace('_', ' ').strip().title()


def metric_count(group):
    return len({m for pairs in DASHBOARDS[group]['pairs'].values()
                for item in pairs for m in (item[1], item[2]) if m})


def percentile_table(frame, pool, group):
    """Every stint in the pool scored on every metric — the notebook's df_pct."""
    cfg = DASHBOARDS[group]
    metrics = []
    for pairs in cfg['pairs'].values():
        for item in pairs:
            metrics += [(m, i == 0) for i, m in enumerate((item[1], item[2])) if m]
    metrics = list(dict.fromkeys(metrics))

    out = pd.DataFrame({
        'player_name': pool['player_name'].map(clean_str),
        'team_name': pool['team_name'].map(clean_str),
        'season_id': pool['season_id'].map(lambda v: '/'.join(as_list(v))),
        'role': pool['position'].map(lambda v: clean_str(v, group)),
    }, index=pool.index)

    for metric, is_quality in metrics:
        if metric not in pool.columns:
            continue
        vals = pool[metric].dropna().astype(float)
        # rank against the pool, matching percentile_of's strictly-less-than share
        pct = pool[metric].map(lambda x: np.nan if pd.isna(x) else (vals < float(x)).mean() * 100)
        if is_quality and cfg['invert'].get(metric, False):
            pct = 100 - pct
        out[f'{metric}__pctile'] = pct
    return out
