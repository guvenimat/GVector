# Ölçüm metodolojisi — bu projeden çıkan dersler

Bu projenin en aktarılabilir çıktısı kod değil, ölçüm disiplini. Aşağıdaki
maddelerin her biri **bu projede yaşanmış bir hatadan** çıktı; örnekler
uydurma değil, `BENCHMARKS.md` ve `DECISIONS.md`'de kayıtlı.

## 1. Ölçüm altyapısı koddan önce gelir

Bir optimizasyonu yazmadan önce onu görebilecek ölçüm hazır olmalı. Aksi
halde "iyileştirdim" ile "iyileştiğini sanıyorum" arasındaki farkı
söyleyemezsiniz. Bu projede `report` binary'si her aşamada büyüdü ve her
kabul kararı ondan çıktı.

## 2. Kabul eşikleri sonuç görülmeden yazılır ve sonradan değiştirilmez

Ön-kayıt kuralı: eşik, onu değerlendirecek ölçüm koşulmadan önce
`DECISIONS.md`'ye yazılır. Karşılanmazsa sonuç **"karşılanmadı" olarak
durur** — eşik yeniden yorumlanmaz.

Bu kural olmasaydı, 9a-2'nin latency eşiği ("50x") ölçümden sonra kolayca
"ama asıl mühürleme penceresi mikrosaniyeye indi, yani geçti sayılır"
haline getirilebilirdi. Getirilmedi: sonuç karşılanmadı olarak durdu,
yanına **kusur kaydı** yazıldı. Kusur kaydı eşiği değiştirmez; eşiğin
ölçmek İSTEDİĞİ şeyle ölçtüğü şey arasındaki farkı belgeler.

Sonucu gördükten sonra alınan kararlar da (ör. ölçüm süresini 2 dakikadan
10 dakikaya çıkarmak) **böyle olduğu açıkça yazılarak** alınır. Okuyucu
kendi indirimini yapsın.

## 3. Eşikler ölçekte ölçülür

Küçük ölçekte geçen bir tasarım büyük ölçekte çökebilir — ve testler yeşil
kalır.

Yaşanmış örnek: ilk backpressure tasarımı birim testlerinde sorunsuzdu,
çünkü küçük ölçekte merge hızlı dönüyor ve denge kuruluyordu. 1M'de
mühürleme ~20 s sürünce denge bozuldu ve **yazıcı 110 saniye boyunca
tamamen durdu** (0 op/s). Testler yeşildi, sistem çalışmıyordu.

## 4. Ölçüm ortamının kendisi kontrol edilir

Uzun süren bir koşunun sonuna eklenen ölçüm, ölçtüğünü sandığınız şeyi
ölçmez.

Yaşanmış örnek: "1M'de okuma ölçeklenmiyor" bulgusu **yanlıştı**. Ölçüm,
5 dakikadır çalışan ve RSS'i 3.1 GB'a çıkmış bir süreçte, uzun bir koşunun
son bölümü olarak alınmıştı. Temiz süreçte f32 okuması 5.4–6.1x
ölçekleniyordu. Düzeltme `DECISIONS.md #44`'te.

Protokol: **taze süreç, warmup, 3 tekrar medyanı, iki ayrı koşuda
doğrulama.**

## 5. Pay ve payda aynı koşulda ölçülür

Bir oran eşiği tanımlıyorsanız, tabanın hangi koşulda ölçüldüğünü eşikle
birlikte yazın.

Yaşanmış örnek: #40'ın taban p99'u (7.8 µs) **fsync'siz** ölçülmüştü. Aynı
eşiği fsync'li bir koşuda değerlendirmek, iyileştirme ne kadar iyi olursa
olsun otomatik başarısızlık üretirdi. Ölçüm koşulu her tekrarında aynı
tutuldu ve bu koşul eşiğin yanına yazıldı.

Aynı ilke veri kümesi için de geçerli: kalıcı bir ölçüm dizini koşular
biriktikçe büyür (1M → 1.64M), yani "aynı ölçüm" artık aynı ölçüm değildir.
Dizin her karşılaştırmadan önce sıfırdan kuruldu.

## 6. Bir kriter başka bir kriterle çelişebilir

Kriterler birbirinden bağımsız yazılır ama sistem bir bütündür.

Yaşanmış örnek: 9a-2'nin kriter 2'sini geçiren mekanizma (backpressure —
kuyruğu sınırlamak için yazıcıyı bekletmek), kriter 1'ini (hiçbir yazma
taban p99'un 50 katından uzun sürmesin) **geçilemez hale getirdi**.

Bu yüzden ön-kayıt kuralına kalıcı bir madde eklendi: eşik yazarken
**"bu kriter, başka hangi kriterle çelişebilir?"** sorusu da sorulur.

## 7. Yarış testleri, yarışın tetiklendiğini de doğrulamalı

Eşzamanlılık testi "hata olmadı" demekle yetinirse, test edilen yarış hiç
oluşmadığında da yeşil kalır — yani sessizce hiçbir şey ölçmez.

Bu projedeki testler yarışın gerçekleştiğini ayrıca iddia eder:
`during_merge > 0`, `seal_in_flight() > 0`, `saw_sealing > 0`,
`max_queue > 0`, `stalls > 0`. Bir testte writer okuyuculardan önce bitince
`iters == 0` oldu ve test **kırıldı** — kırılması doğruydu, çünkü ölçtüğü
şey artık oluşmuyordu.

## 8. Sapmanın yönünü önceden düşün

Bir ölçüm yönteminin hatası varsa, hangi yönde saptığını ölçümden önce
belirleyin: hangi sonuç güvenli, hangisi şüpheli olur?

Yaşanmış örnek: metadata bellek tahminini doğrularken yapılar bırakılıp RSS
farkı ölçüldü. Serbest bırakılan bellek işletim sistemine hemen dönmeyebilir
→ gerçek düşüş **olduğundan az** görünür → bu sapma "tahmin şişkin"
sonucunu güvenli, "tahmin doğru" sonucunu şüpheli kılar. Sonuç güvenli
tarafa düştü (tahmin gerçeği eksik gösteriyordu), dolayısıyla karar
sağlamdı.

## 9. Ham rakamı ve normalizasyonu birlikte göster

Normalize edilmiş bir sayıyı ham gibi sunmak, sonradan tekrar ölçen birini
şaşırtır. "Ölçüldü: 1.5M'de X, 1M eşdeğeri: Y" biçimi hem doğrulanabilir
hem karşılaştırılabilir.

## 10. Ölçüm çıktısını filtreleyerek okuma

`grep`'lenen bir çıktı panic'i gizler ve pipe `exit 0` döndürür. Bu projede
bir ölçüm modu `DuplicateId` ile çöküyordu ve iki koşu boyunca fark
edilmedi. Çıktı filtresiz okunur; çıkış kodu ayrıca kontrol edilir.
