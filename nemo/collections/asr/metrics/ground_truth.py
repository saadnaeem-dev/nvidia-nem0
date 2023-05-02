from nemo.collections.asr.metrics.wer import word_error_rate
import pandas as pd
reference = [
    "CHAPTER FIVE THE SEAL AND THE BEAR YOU KNOW DOCTOR SAID HATTERAS AS THEY RETURNED TO THE HUT THE POLAR BEARS SUBSIST ALMOST ENTIRELY ON SEALS THEY'LL LIE IN WAIT FOR THEM BESIDE THE CREVASSES FOR WHOLE DAYS. READY TO STRANGLE THEM THE MOMENT THEIR HEADS APPEAR ABOVE THE SURFACE IT IS NOT LIKELY THEN THAT A BEAR WILL BE FRIGHTENED OF A SEAL I THINK I SEE WHAT YOU ARE AFTER BUT IT IS DANGEROUS. YES BUT THERE IS MORE CHANCE OF SUCCESS THAN IN TRYING ANY OTHER PLAN SO I MEAN TO RISK IT I AM GOING TO DRESS MYSELF IN THE SEAL'S SKIN AND CREEP ALONG THE ICE COME DON'T LET US LOSE TIME LOAD THE GUN AND GIVE IT ME. THE DOCTOR COULD NOT SAY ANYTHING FOR HE WOULD HAVE DONE THE SAME HIMSELF SO HE FOLLOWED HATTERAS SILENTLY TO THE SLEDGE TAKING WITH HIM A COUPLE OF HATCHETS FOR HIS OWN AND JOHNSON'S USE HATTERAS SOON MADE HIS TOILETTE AND SLIPPED INTO THE SKIN. NOW THEN GIVE ME THE GUN HE SAID. COURAGE HATTERAS SAID THE DOCTOR HANDING HIM THE WEAPON WHICH HE HAD CAREFULLY LOADED MEANWHILE NEVER FEAR BUT BE SURE YOU DON'T SHOW YOURSELVES TILL I FIRE THE DOCTOR SOON JOINED THE OLD BOATSWAIN BEHIND THE HUMMOCK AND TOLD HIM WHAT THEY HAD BEEN DOING. THE BEAR WAS STILL THERE BUT MOVING RESTLESSLY ABOUT AS IF HE FELT THE APPROACH OF DANGER IN A QUARTER OF AN HOUR OR SO THE SEAL MADE HIS APPEARANCE ON THE ICE HE HAD GONE A GOOD WAY ROUND SO AS TO COME ON THE BEAR BY SURPRISE. AND EVERY MOVEMENT WAS SO PERFECT AN IMITATION OF A SEAL THAT EVEN THE DOCTOR WOULD HAVE BEEN DECEIVED IF HE HAD NOT KNOWN IT WAS HATTERAS IT IS CAPITAL SAID JOHNSON IN A LOW VOICE THE BEAR HAD INSTANTLY CAUGHT SIGHT OF THE SUPPOSED SEAL FOR HE GATHERED HIMSELF UP. PREPARING TO MAKE A SPRING AS THE ANIMAL CAME NEARER. BRUIN WENT TO WORK WITH EXTREME PRUDENCE THOUGH HIS EYES GLARED WITH GREEDY DESIRE TO CLUTCH THE COVETED PREY FOR HE HAD PROBABLY BEEN FASTING A MONTH IF NOT TWO HE ALLOWED HIS VICTIM TO GET WITHIN TEN PACES OF HIM AND THEN SPRANG FORWARD WITH A TREMENDOUS BOUND. BUT STOPPED SHORT STUPEFIED AND FRIGHTENED WITHIN THREE STEPS OF HATTERAS WHO STARTED UP THAT MOMENT AND THROWING OFF HIS DISGUISE KNELT ON ONE KNEE AND AIMED STRAIGHT AT THE BEAR'S HEART. HURRYING TOWARDS HATTERAS FOR THE BEAR HAD REARED ON HIS HIND LEGS AND WAS STRIKING THE AIR WITH ONE PAW AND TEARING UP THE SNOW TO STANCH HIS WOUND WITH THE OTHER HATTERAS NEVER MOVED BUT WAITED KNIFE IN HAND. HE HAD AIMED WELL AND FIRED WITH A SURE AND STEADY AIM BEFORE EITHER OF HIS COMPANIONS CAME UP HE HAD PLUNGED THE KNIFE IN THE ANIMAL'S THROAT AND MADE AN END OF HIM FOR HE FELL DOWN AT ONCE TO RISE NO MORE."
]
concatenated_string = " ".join(reference)
reference = [concatenated_string]

model_names=[]


# --------- stt_en_fastconformer_transducer_large_ls--------------------
model_name = "stt_en_fastconformer_transducer_large_ls"
dict_list = [
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0000.wav", "pred_text": "chapter five the seal and the bear you know doctor said hatteras as they returned to the hut the polar bears subsist almost entirely on seals they'll lie in wait for them beside the crevasses for whole days"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0001.wav", "pred_text": "ready to strangle them the moment their heads appear above the surface it is not likely then that a bear will be frightened of a seal i think i see what you are after but it is dangerous"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0002.wav", "pred_text": "yes but there is more chance of success than in trying any other plan so i mean to risk it i am going to dress myself in the seal's skin and creep along the ice come don't let us lose time load the gun and give it me"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0003.wav", "pred_text": "the doctor could not say anything for he would have done the same himself so he followed hatteras silently to the sledge taking with him a couple of hatchets for his own and johnson's use hatteras soon made his toilette and slipped into the skin"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0004.wav", "pred_text": "now then give me the gun he said"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0005.wav", "pred_text": "courage hatteras said the doctor handing him the weapon which he had carefully loaded meanwhile never fear but be sure you don't show yourselves till i fire the doctor soon joined the old boatswain behind the hummock and told him what they had been doing"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0006.wav", "pred_text": "the bear was still there but moving restlessly about as if he felt the approach of danger in a quarter of an hour or so the seal made his appearance on the ice he had gone a good way round so as to come on the bear by surprise"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0007.wav", "pred_text": "and every movement was so perfect an imitation of a seal that even the doctor would have been deceived if he had not known it was hatteras it is capital said johnson in a low voice the bear had instantly caught sight of the supposed seal for he gathered himself up"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0008.wav", "pred_text": "preparing to make a spring as the animal came nearer"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0009.wav", "pred_text": "bruin went to work with extreme prudence though his eyes glared with greedy desire to clutch the coveted prey for he had probably been fasting a month if not two he allowed his victim to get within ten paces of him and then sprang forward with a tremendous bound"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0010.wav", "pred_text": "but stopped short stupefied and frightened within three steps of hatteras who started up that moment and throwing off his disguise knelt on one knee and aimed straight at the bear's heart"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0011.wav", "pred_text": "hurrying towards hatteras for the bear had reared on his hind legs and was striking the air with one paw and tearing up the snow to stanch his wound with the other hatteras never moved but waited knife in hand"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0012.wav", "pred_text": "he had aimed well and fired with a sure and steady aim before either of his companions came up he had plunged the knife in the animal's throat and made an end of him for he fell down at once to rise no more"}

]
stt_en_fastconformer_transducer_large_ls_hypothesis = [d["pred_text"] for d in dict_list]

concatenated_string = " ".join(stt_en_fastconformer_transducer_large_ls_hypothesis)
model_text_name = [[concatenated_string],model_name]
model_names.append(model_text_name)
execution_times = []
stt_en_fastconformer_transducer_large_ls = ('GPU', 5.918583869934082)
execution_times.append(stt_en_fastconformer_transducer_large_ls)

# --------- stt_en_squeezeformer_ctc_medium_ls--------------------
model_name = "stt_en_squeezeformer_ctc_medium_ls"
dict_list = [
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0000.wav", "pred_text": "chapter five the seal and the bear you know doctor said hatteras as they returned to the hut the polar bears subsist almost entirely on seals they'll lie in wait for them beside the crevasss for whole days"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0001.wav", "pred_text": "ready to strangle them the moment their heads appear above the surface it is not likely then that a bear will be frightened of a seal i think i see what you are after but it is dangerous"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0002.wav", "pred_text": "yes but there is more chance of success than in trying any other plan so i mean to risk it i am going to dress myself in the seal's skin and creep along the ice come don't let us lose time load the gun and give it me"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0003.wav", "pred_text": "the doctor could not say anything for he would have done the same himself so he followed hatteras silently to the sledge taking with him a couple of hatchets for his own and johnson's use hatteras soon made his toilet and slipped into the skin"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0004.wav", "pred_text": "now then give me the gun he said"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0005.wav", "pred_text": "courage hatteras said the doctor handing him the weapon which he had carefully loaded meanwhile never fear but be sure you don't show yourselves till i fire the doctor soon joined the old boatswain behind the hummock and told him what they had been doing"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0006.wav", "pred_text": "the bear was still there but moving restlessly about as if he felt the approach of danger in a quarter of an hour or so the seal made his appearance on the ice he had gone a good way round so as to come on the bear by surprise"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0007.wav", "pred_text": "and every movement was so perfect an imitation of a seal that even the doctor would have been deceived if he had not known it was hatteras it is capital said johnson in a low voice the bear had instantly caught sight of the supposed seal for he gathered himself up"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0008.wav", "pred_text": "preparing to make a spring as the animal came nearer"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0009.wav", "pred_text": "bruin went to work with extreme prudence though his eyes glared with greedy desire to clutch the coveted prey for he had probably been fasting a month if not two he allowed his victim to get within ten paces of him and then sprang forward with a tremendous bound"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0010.wav", "pred_text": "but stopped short stupefied and frightened within three steps of hatteras who started upt that moment and throwing off his disguise knelt on one knee and aimed straight at the bear's heart"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0011.wav", "pred_text": "hurrying towards hatteras for the bear had reared on his hind legs and was striking the air with one paw and tearing up the snow to stanch his wound with the other hatteras never moved but waited knife in hand"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0012.wav", "pred_text": "he had aimed well and fired with a sure and steady aim before either of his companions came up he had plunged the knife into the animal's throat and made an end of him for he fell down at once to rise no more"},

]
concatenated_pred_text = ' '.join(d['pred_text'] for d in dict_list)
model_text_name = [[concatenated_string], model_name]
model_names.append(model_text_name)
stt_en_squeezeformer_ctc_medium_ls = ('GPU', 4.086786270141602)
execution_times.append(stt_en_squeezeformer_ctc_medium_ls)

# --------- stt_en_squeezeformer_ctc_small_medium_ls--------------------
model_name = "stt_en_squeezeformer_ctc_small_medium_ls"
dict_list = [
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0000.wav", "pred_text": "chapter five the seal and the bear you know doctor said hatteras as they returned to the hut the polar bears subsist almost entirely on seals they'll lie in wait for them beside the crevasses for whole days"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0001.wav", "pred_text": "ready to strangle them the moment their heads appear above the surface it is not likely then that a bear will be frightened of a seal i think i see what you are after but it is dangerous"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0002.wav", "pred_text": "yes but there is more chance of success than in trying any other plan so i mean to risk it i am going to dress myself in the seal's skin and creep along the ice come don't let us lose time load the gun and give it to me"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0003.wav", "pred_text": "the doctor could not say anything for he would have done the same himself so he followed hatteras silently to the sledge taking with him a couple of hatchets for his own and johnson's use hatteras soon made his toilette and slipped into the skin"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0004.wav", "pred_text": "now then give me the gun he said"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0005.wav", "pred_text": "courage hatteras said the doctor handing him the weapon which he had carefully loaded meanwhile never fear but be sure you don't show yourselves till i fire the doctor soon joined the old boatswain behind the hummock and told him what they had been doing"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0006.wav", "pred_text": "the bear was still there but moving restlessly about as if he felt the approach of danger in a quarter of an hour or so the seal made his appearance on the ice he had gone a good way round so as to come on the bear by surprise"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0007.wav", "pred_text": "and every movement was so perfect an imitation of a seal that even the doctor would have been deceived if he had not known it was hatteras it is capital said johnson in a low voice the bear had instantly caught sight of the supposed seal for he gathered himself up"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0008.wav", "pred_text": "preparing to make a spring as the animal came nearer"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0009.wav", "pred_text": "bruin went to work with extreme prudence though his eyes glared with greedy desire to clutch the coveted prey for he had probably been fasting a month if not two he allowed his victim to get within ten paces of him and then sprang forward with a tremendous bound"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0010.wav", "pred_text": "but stopped short stupefied and frightened within three steps of hatteras who started up at that moment and throwing off his disguise knelt on one knee and aimed straight at the bear's heart"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0011.wav", "pred_text": "hurrying towards hatteras for the bear had reared on his hind legs and was striking the air with one paw and tearing up the snow to stanch his wound with the other hatteras never moved but waited knife in hand"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0012.wav", "pred_text": "he had aimed well and fired with a sure and steady aim before either of his companions came up he had plunged the knife into the animal's throat and made an end of him for he fell down at once to rise no more"},

]
concatenated_pred_text = ' '.join(d['pred_text'] for d in dict_list)
model_text_name = [[concatenated_string], model_name]
model_names.append(model_text_name)
stt_en_squeezeformer_ctc_small_medium_ls = ('GPU', 3.4562795162200928)
execution_times.append(stt_en_squeezeformer_ctc_small_medium_ls)


# --------- stt_en_squeezeformer_ctc_xsmall_ls--------------------
model_name = "stt_en_squeezeformer_ctc_xsmall_ls"
dict_list = [

{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0000.wav", "pred_text": "chapter five the seal and the bear you know doctor said hatteras as they returned to the hut the polar bearsubsist almost entirely on seals they'll lie and wait for them beside the crevices for whole days"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0001.wav", "pred_text": "ready to strangle them the moment their heads appear above the surface it is not likely then that a bear will be frightened of a seal i think i see what you are after but it is dangerous"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0002.wav", "pred_text": "yes but there is more chance of success than in trying any other plan so i mean to risk it i am going to dress myself in the seal's skin and creep along the ice come don't let us lose time load the gun and give it to me"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0003.wav", "pred_text": "the doctor could not say anything for he would have done the same himself so he followed hatteras silently to the sledge taking with him a couple of hatchets for his own in johnson's use hatteras soon made his toilette and slipped into the skin"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0004.wav", "pred_text": "now then give me the gun he said"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0005.wav", "pred_text": "courage hatteras said the doctor handing him the weapon which he had carefully loaded meanwhile never fear but be sure you don't show yourselves till i fire the doctor soon joined the old boat swain behind the hummock and told him what they had been doing"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0006.wav", "pred_text": "the bear was still there but moving restlessly about as if he felt the approach of danger in a quarter of an hour or so the seal made his appearance on the ice he had gone a good way round so as to come on the bear by surprise"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0007.wav", "pred_text": "and every movement was so perfect an imitation of a seal that even the doctor would have been deceived if he had not known it was hatteras it is capital said johnson in a low voice the bear had instantly caught sight of the supposed seal for he gathered himself up"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0008.wav", "pred_text": "preparing to make a spring as the animal came nearer"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0009.wav", "pred_text": "brewin went to work with extreme prudence though his eyes glared with greedy desire to clutch the coveted prey for he had probably been fasting a month if not to he allowed his victim to get within ten paces of him and then sprang forward with a tremendous bound"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0010.wav", "pred_text": "but stopped short stupefied and frightened within three steps of hatteras who started up at that moment and throwing off his disguise knelt on one knee and aimed straight at the bear's heart"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0011.wav", "pred_text": "hurrying towards hatteras for the bear had reared on his hind legs and was striking the air with one paw and tearing up the snow to stanch his wound with the other hatteras never moved but waited knife in hand"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0012.wav", "pred_text": "he had aimed well and fired with a sure and steady aim before either of his companions came up he had plunged the knife into the animal's throat and made an end of him for he fell down at once to rise no more"},

]
concatenated_pred_text = ' '.join(d['pred_text'] for d in dict_list)
model_text_name = [[concatenated_string], model_name]
model_names.append(model_text_name)
stt_en_squeezeformer_ctc_xsmall_ls = ('GPU', 3.1957037448883057)
execution_times.append(stt_en_squeezeformer_ctc_xsmall_ls)


# --------- SttEnFastConformerCtcLarge--------------------
model_name = "SttEnFastConformerCtcLarge"
dict_list = [

{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0000.wav", "pred_text": "chapter five the seal and the bear you know doctor said hatteras as they returned to the hut the polar bears subsist almost entirely on seals they'll lie and wait for them beside the crevices for whole days"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0001.wav", "pred_text": "ready to strangle them the moment their heads appear above the surface it is not likely then that a bear will be frightened of a seal i think i see what you are after but it is dangerous"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0002.wav", "pred_text": "yes but there is more chance of success than in trying any other plan so i mean to risk it i am going to dress myself in the seal's skin and creep along the ice come don't let us lose time load the gun and give it to me"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0003.wav", "pred_text": "the doctor could not say anything for he would have done the same himself so he followed hatteras silently to the sledge taking with him a couple of hatchets for his own and johnson's use hatteras soon made his toilet and slipped into the skin"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0004.wav", "pred_text": "now then give me the gun he said"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0005.wav", "pred_text": "courage hatteras said the doctor handing him the weapon which he had carefully loaded meanwhile never fear but be sure you don't show yourselves till i fire the doctor soon joined the old boatswain behind the hummock and told him what they had been doing"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0006.wav", "pred_text": "the bear was still there but moving restlessly about as if he felt the approach of danger in a quarter of an hour or so the seal made his appearance on the ice he had gone a good way round so as to come on the bear by surprise"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0007.wav", "pred_text": "and every movement was so perfect an imitation of a seal that even the doctor would have been deceived if he had not known it was hatteras it is capital said johnson in a low voice the bear had instantly caught sight of the supposed seal for he gathered himself up"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0008.wav", "pred_text": "preparing to make a spring as the animal came nearer"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0009.wav", "pred_text": "bruin went to work with extreme prudence though his eyes glared with greedy desire to clutch the coveted prey for he had probably been fasting a month if not two he allowed his victim to get within ten paces of him and then sprang forward with a tremendous bound"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0010.wav", "pred_text": "but stopped short stupefied and frightened within three steps of hatteras who started up at that moment and throwing off his disguise knelt on one knee and aimed straight at the bear's heart"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0011.wav", "pred_text": "hurrying towards hatteras for the bear had reared on his hind legs and was striking the air with one paw and tearing up the snow to stanch his wound with the other hatteras never moved but waited knife in hand"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0012.wav", "pred_text": "he had aimed well and fired with a sure and steady aim before either of his companions came up he had plunged the knife into the animal's throat and made an end of him for he fell down at once to rise no more"},

]
concatenated_pred_text = ' '.join(d['pred_text'] for d in dict_list)
model_text_name = [[concatenated_string], model_name]
model_names.append(model_text_name)
SttEnFastConformerCtcLarge = ('GPU', 8.587157487869263)
execution_times.append(SttEnFastConformerCtcLarge)

# --------- SttEnFastConformerTransducerLarge--------------------
model_name = "SttEnFastConformerTransducerLarge"
dict_list = [

{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0000.wav", "pred_text": "chapter five the seal and the bear you know doctor said hatteras as they returned to the hut the polar bears subsist almost entirely on seals they'll lie in wait for them beside the crevices for whole days"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0001.wav", "pred_text": "ready to strangle them the moment their heads appear above the surface it is not likely then that a bear will be frightened of a seal i think i see what you are after but it is dangerous"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0002.wav", "pred_text": "yes but there is more chance of success than in trying any other plan so i mean to risk it i am going to dress myself in the seal's skin and creep along the ice come don't let us lose time load the gun and give it to me"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0003.wav", "pred_text": "the doctor could not say anything for he would have done the same himself so he followed hatteras silently to the sledge taking with him a couple of hatchets for his own and johnson's use hatteras soon made his toilet and slipped into the skin"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0004.wav", "pred_text": "now then give me the gun he said"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0005.wav", "pred_text": "courage hatteras said the doctor handing him the weapon which he had carefully loaded meanwhile never fear but be sure you don't show yourselves till i fire the doctor soon joined the old boatswain behind the hummock and told him what they had been doing"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0006.wav", "pred_text": "the bear was still there but moving restlessly about as if he felt the approach of danger in a quarter of an hour or so the seal made his appearance on the ice he had gone a good way round so as to come on the bear by surprise"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0007.wav", "pred_text": "and every movement was so perfect in imitation of a seal that even the doctor would have been deceived if he had not known it was hatteras it is capital said johnson in a low voice the bear had instantly caught sight of the supposed seal for he gathered himself up"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0008.wav", "pred_text": "preparing to make a spring as the animal came nearer"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0009.wav", "pred_text": "bruin went to work with extreme prudence though his eyes glared with greedy desire to clutch the coveted prey for he had probably been fasting a month if not two he allowed his victim to get within ten paces of him and then sprang forward with a tremendous bound"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0010.wav", "pred_text": "but stopped short stupefied and frightened within three steps of hatteras who started up at that moment and throwing off his disguise knelt on one knee and aimed straight at the bear's heart"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0011.wav", "pred_text": "hurrying towards hatteras for the bear had reared on his hind legs and was striking the air with one paw and tearing up the snow to stanch his wound with the other hatteras never moved but waited knife in hand"},
{"audio_filepath": "C:\\Users\\saadn\\PycharmProjects\\DATA\\295000_wav_files\\8765-295000-0012.wav", "pred_text": "he had aimed well and fired with a sure and steady aim before either of his companions came up he had plunged the knife into the animal's throat and made an end of him for he fell down at once to rise no more"},

]
concatenated_pred_text = ' '.join(d['pred_text'] for d in dict_list)
model_text_name = [[concatenated_string], model_name]
model_names.append(model_text_name)
SttEnFastConformerTransducerLarge = ('GPU', 7.846684455871582)
execution_times.append(SttEnFastConformerTransducerLarge)


# create a dictionary and append it with model name and its wer then create a pandas dataframe with two columns model name and wer and save it to the csv file
results = {}
wers = []
ref = []
hyp=[]
m_name = []
device = []
exec_time = []

def wer(reference, hypothesis):
    # Tokenize the input strings
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()

    # Initialize the edit distance matrix
    d = [[0] * (len(hyp_tokens) + 1) for _ in range(len(ref_tokens) + 1)]

    # Populate the first row and column
    for i in range(len(ref_tokens) + 1):
        d[i][0] = i
    for j in range(len(hyp_tokens) + 1):
        d[0][j] = j

    # Calculate the edit distance matrix
    for i in range(1, len(ref_tokens) + 1):
        for j in range(1, len(hyp_tokens) + 1):
            substitution_cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,  # deletion
                d[i][j - 1] + 1,  # insertion
                d[i - 1][j - 1] + substitution_cost,  # substitution
            )

    # Calculate the WER
    wer = d[len(ref_tokens)][len(hyp_tokens)] / len(ref_tokens)
    return wer


for model_name, device_exec_time in zip(model_names, execution_times):
    # wer.append(word_error_rate(hypotheses=model_name[0], references=reference))
    wers.append(wer(reference[0], model_name[0][0]))
    m_name.append(model_name[1])
    device.append(device_exec_time[0])
    exec_time.append(device_exec_time[1])
    hyp.append(model_name[0])
    ref.append(reference)

results['model_name'] = m_name
results['wer'] = wer
# results['hyp'] = hyp
# results['ref'] = ref
results['device'] = device
results['exec_time'] = exec_time
# create a pandas dataframe using results and save it to csv file
df = pd.DataFrame(results)
df.to_csv('STT_Libri_Experiments_Analysis.csv', index=False)
